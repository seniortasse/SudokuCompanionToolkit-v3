// ==============================================================================
// MainActivity — CameraX pipeline, gating, rectification handoff, and capture UX
// ==============================================================================
// WHY
// ------------------------------------------------------------------------------
// We need a smooth capture loop that finds a Sudoku grid in real time, validates it with corner heatmaps, and only locks/captures when everything is clearly good. MainActivity coordinates that flow and keeps the UI responsive.
//
// WHAT
// ------------------------------------------------------------------------------
// Sets up CameraX (Preview + ImageAnalysis), throttles frames, runs Detector and CornerRefiner, draws the HUD via OverlayView, enforces multiple gates (peaks, geometry, area, aspect, jitter, cyan-guard), and only then rectifies + classifies. Success triggers a shutter effect and navigates to results.
//
// HOW (high-level)
// ------------------------------------------------------------------------------
// 1) Request CAMERA permission; start CameraX with a Preview and an RGBA ImageAnalysis stream.
// 2) For every analyzed frame (throttled), convert to Bitmap.
// 3) Detector.infer() -> pick the most-centered box.
// 4) CornerRefiner.refine() on that ROI -> peaks and TL/TR/BR/BL.
// 5) OverlayView updates HUD (boxes, peaks, cyan guard, optional green ROI).
// 6) If N consecutive frames pass all gates, try rectification + classification; upon success, lock & celebrate.
//
// FILE ORGANIZATION
// ------------------------------------------------------------------------------
// • Fields: camera views, ML components (Detector, CornerRefiner, DigitClassifier), gates, histories.
// • Lifecycle: onCreate() -> startCamera(); onDestroy() cleanup.
// • Analyzer: throttling logic, detection + refine + HUD update + gating.
// • Handoff: attemptRectifyAndClassify() (best-of-N capture) and onLockedGridCaptured().
// • Geometry helpers: area/aspect/convexity/jitter/guard mapping.
//
// RUNTIME FLOW / HOW THIS FILE IS USED
// ------------------------------------------------------------------------------
// User points camera; cyan guard guides aim. Detector finds grids; we choose the most centered. CornerRefiner validates corners. Passing frames accumulate; on success we rectify + classify, play shutter, and show results. Otherwise we keep scanning — no premature locks.
//
// NOTES
// ------------------------------------------------------------------------------
// Comments are ASCII-only. Original code untouched; just comment lines added. Look for 'gates' in analyzer to see the criteria used before capture.
//
// ==============================================================================
package com.contextionary.sudoku


import com.contextionary.sudoku.logic.SudokuSolver
import com.contextionary.sudoku.logic.SudokuAutoCorrector
import com.contextionary.sudoku.logic.GridPrediction
import com.contextionary.sudoku.logic.AutoCorrectionResult


import com.contextionary.sudoku.logic.GridState
import com.contextionary.sudoku.logic.GridCellState
import com.contextionary.sudoku.logic.SudokuGrid

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Size as UiSize
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

// === OpenCV ===
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size as CvSize   // <-- use alias
import org.opencv.imgproc.Imgproc

import android.os.Handler
import android.os.Looper


import com.contextionary.sudoku.telemetry.ConversationTelemetry


import android.graphics.Rect
import android.graphics.PointF

import android.util.TypedValue

import com.contextionary.sudoku.logic.GridConversationCoordinator
import android.view.View
import android.view.ViewGroup
import android.view.Gravity
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Color








import com.contextionary.sudoku.profile.UserProfileStore
import com.contextionary.sudoku.profile.toSnapshot



import com.google.android.material.button.MaterialButton



import androidx.appcompat.view.ContextThemeWrapper


import com.contextionary.sudoku.logic.*
import com.contextionary.sudoku.logic.SudokuLLMConversationCoordinator
import com.contextionary.sudoku.logic.SudokuLLMClient
import com.contextionary.sudoku.logic.FakeSudokuLLMClient
import com.contextionary.sudoku.logic.RealSudokuLLMClient

import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

import android.text.TextUtils

import android.speech.tts.TextToSpeech


import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager


import com.contextionary.sudoku.conversation.InMemoryTurnStore
import com.contextionary.sudoku.conversation.TurnLifecycleManager
import com.contextionary.sudoku.conversation.PromptBuilder
import com.contextionary.sudoku.conversation.RecoveryController
import com.contextionary.sudoku.conversation.PersonaDescriptor
import com.contextionary.sudoku.conversation.TurnStore





import com.contextionary.sudoku.conductor.SudoStore
import com.contextionary.sudoku.conductor.policy.CoordinatorPolicyAdapter


import java.security.MessageDigest

import com.contextionary.sudoku.logic.LlmToolCall

import com.contextionary.sudoku.logic.GridSeverity
import com.contextionary.sudoku.logic.computeSeverity


private val manualEditLog = mutableListOf<LLMCellEditEvent>()
private var manualEditSeq = 0





private fun sha256Hex(s: String): String {
    val md = MessageDigest.getInstance("SHA-256")
    val bytes = md.digest(s.toByteArray(Charsets.UTF_8))
    return bytes.joinToString("") { "%02x".format(it) }
}


// File-level so any helper functions in MainActivity.kt can see it.
data class Solvability(
    val hasNoSolution: Boolean,
    val solutionCountCapped: Int
)







// TODO: move this to BuildConfig or secure storage before shipping!

// The central orchestrator for camera preview, analysis, HUD, detection, corner refinement, gating, and the final capture/rectify/classify path.
class MainActivity : ComponentActivity() {


    // --- Sudo TTS state ---
    private lateinit var tts: TextToSpeech
    private var ttsReady: Boolean = false
    private var lastUtteranceId: String? = null

    private val ttsPending: MutableList<String> = mutableListOf()

    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var detector: Detector
    private lateinit var analyzerExecutor: ExecutorService
    private lateinit var shutter: android.media.MediaActionSound

    // Removed the duplicate lateinit version; keep this one:
    //private var digitClassifier: DigitClassifier? = null
    private var handoffInProgress = false

    private var resultsSudokuView: SudokuResultView? = null

    // ==== Display state for SudokuResultView (what the user sees) ====
    // These drive font choice (printed vs handwritten) and candidates.
    private var uiDigits = IntArray(81)          // final digit shown in each cell
    private var uiConfs  = FloatArray(81)        // confidence for the shown digit
    private var uiGiven  = BooleanArray(81)      // true → render with "printed" font
    private var uiSol    = BooleanArray(81)      // true → render with "handwritten" font
    private var uiAuto   = BooleanArray(81)      // true → render printed BOLD (autocorrected)
    private var uiCand   = IntArray(81)          // 9-bit candidate mask per cell

    //private val uiManual = BooleanArray(81) { false }  // NEW: manual-corrected
    private val uiManual = BooleanArray(81) { false }        // manual corrected?
    private val uiManualFrom = IntArray(81) { 0 }            // last manual from digit
    private val uiManualTo   = IntArray(81) { 0 }            // last manual to digit
    private val uiManualSeq  = IntArray(81) { 0 }

    private val uiConfirmed = BooleanArray(81)
    // per-cell edit seq (optional)
    private var manualEditSeqGlobal = 0

    // Small helper to clear UI state between captures
    private fun resetUiArrays() {
        java.util.Arrays.fill(uiDigits, 0)
        java.util.Arrays.fill(uiConfs, 1.0f)
        java.util.Arrays.fill(uiGiven, false)
        java.util.Arrays.fill(uiSol, false)
        java.util.Arrays.fill(uiAuto, false)
        java.util.Arrays.fill(uiCand, 0)
        //java.util.Arrays.fill(uiManual, false)
        java.util.Arrays.fill(uiManual, false)
        java.util.Arrays.fill(uiManualFrom, 0)
        java.util.Arrays.fill(uiManualTo, 0)
        java.util.Arrays.fill(uiManualSeq, 0)
        manualEditSeqGlobal = 0
    }

    // === Sudoku logic (solver + auto-corrector) ===
    private val sudokuSolver = SudokuSolver()
    private val sudokuAutoCorrector = SudokuAutoCorrector(sudokuSolver)
    private var lastGridPrediction: GridPrediction? = null
    private var lastAutoCorrectionResult: AutoCorrectionResult? = null

    // === LLM conversation orchestration for the result grid ===
    private val gridConversationCoordinator = GridConversationCoordinator()


    // Keep the full 9×9 readouts so the view can render printed/handwritten/candidates
    private var lastCellReadouts: Array<Array<CellReadout>>? = null



    // === Timing / throttling ===
    private var frameIndex = 0
    private var lastInferMs = 0L
    private val minInferIntervalMs = 80L
    private val skipEvery = 0



    private val speakLock = Any()
    @Volatile private var speakInFlight: Boolean = false
    @Volatile private var queuedSpeak: Pair<String, Boolean>? = null

    private var nextSpeakReqId: Int = 1




    // --- Voice/ASR state ---
    private val mainHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private var isSpeaking: Boolean = false          // true while any TTS is playing
    private var asrActive: Boolean = false           // true while SpeechRecognizer is listening


    private val turnController = ConversationTurnController()


    // When true, startShortListening() will run after the current utterance finishes.
    //private var wantListenAfterUtterance: Boolean = false


    // When this utterance finishes, immediately start listening
    private var pendingListenAfterUtteranceId: String? = null


    //private lateinit var asr: SudoASR

    private var asr: SudoASR? = null


    private val USE_CONDUCTOR_ONLY = true


    // --- Conversation turn persistence & lifecycle (do NOT name anything "lifecycle" in a ComponentActivity) ---
    //private val turnStore = InMemoryTurnStore()

    private val turnStore: TurnStore by lazy(LazyThreadSafetyMode.NONE) { InMemoryTurnStore() }

    //private val turnLifecycle: TurnLifecycleManager by lazy(LazyThreadSafetyMode.NONE) { TurnLifecycleManager(turnStore) }





    private lateinit var sudoStore: SudoStore



    private val turnLifecycle by lazy(LazyThreadSafetyMode.NONE) {
        TurnLifecycleManager(store = turnStore)
    }

    private val promptBuilder by lazy(LazyThreadSafetyMode.NONE) {
        PromptBuilder(store = turnStore)
    }

    private val recovery by lazy(LazyThreadSafetyMode.NONE) {
        RecoveryController(store = turnStore, lifecycle = turnLifecycle)
    }

    //private val recovery = RecoveryController()

    private val systemPromptText = """

PERSONA — “Witty Sidekick” (Friend-half of Sudo DNA)

You are “Sudo,” a funny Sudoku-solving coach: a witty sidekick who helps the user improve at Sudoku through clear steps, specific feedback, and motivating accountability. The user is the protagonist; you are the coach.

Tone: balanced and professional with light wit (level 3). Tiny snark only: gentle teasing is allowed only if obviously affectionate and never when the user is frustrated or confused. No emojis.

Here is the style of humour you should use:

“‘Hidden single’ is called ‘hidden’ because the cell looks empty, but the digit is the one that’s cornered.”
“Scan box 3 for a hidden single: if a digit has only one legal home, it’s moving in. No roommates, no drama.”
“Your pencil marks are… ambitious. Let’s prune them before they form a small government.”
“We’re not guessing. That’s how Sudoku turns into soap opera.”

Opening statements (start of session) should usually look as below (Do not just say "Hey!" - Be colorful and fun from first interaction).
Use a SIMILAR humour without being a copycat
 (these are just random examples, not meant
to be copied but rather to get inspiration from):
“Alright, coach hat on. Let’s make this grid behave.”
“Okay, Sudoku time. I brought logic and a tiny whistle.”
“Welcome back. We’re about to turn confusion into coordinates.”
“Let’s do this: clean scan, sharp moves, zero drama.”
“I’m here, the grid is here, and guessing is not invited.”
“Good. We’ll solve this the polite way: one deduction at a time.”
“Alright—eyes on the grid. We hunt patterns, not miracles.”
“Let’s warm up: we’ll start with the easy wins hiding in plain sight.”
“Coach mode activated. I’ll guide; you’ll land the punches.”
“Today’s agenda: fewer candidates, more certainty.”
“Okay, show me the battlefield—rows, columns, and suspicious blanks.”
“We’re about to make nine tiny neighborhoods follow the rules.”


Mission: Make the user better at Sudoku and help them solve the current puzzle efficiently. Always prioritize progress:
Give a concrete next action first (what to scan, what to write, what to check).
Then explain briefly why it works.
Ask one targeted question or offer an A/B choice to keep momentum.

Curious-facts drip (required):
Include short interesting facts when appropriate that relate to Sudoku (history, rules, techniques, common pitfalls, naming, logic principles) and optionally 1 adjacent fact that supports learning (pattern recognition, cognition, habits, error-checking), but only if clearly connected.
No trivia dumps: facts must be useful, memorable, and tied to what we’re doing now.
Reduce or pause facts if the user is overwhelmed; prioritize clarity and calm.
DO NOT announce them with phrases like "fun fact" or "Sudoku tip" or similar titles or
headers. Just naturally blend them in your conversation, in a subtle way, that
keeps the flow of the conversation without sounding like changing the focus of your talk.

Fun facts would sometimes look as below (these are just random examples, not meant
to be copied but rather to get inspiration from):
“Sudoku isn’t about math; it’s a constraint-satisfaction puzzle. That’s why ‘candidate elimination’ beats ‘calculation.’”
“This is the same mental skill used in debugging: reduce the search space until only one option survives.”
“Sudoku isn’t math; it’s constraint logic. The numbers are just symbols.”
“Most human-solving is pattern recognition dressed up as logic.”
“A ‘hidden single’ is ‘hidden’ because the cell isn’t obvious—the digit is forced.”
“Locked candidates are basically ‘box politics’: a digit is confined to a line, so it can’t appear elsewhere on that line.”
“Good solvers spend more time eliminating than placing.”
“Many published Sudokus are designed so you never need guessing—just the right technique ladder.”
“Candidate notation is a memory aid, not a requirement. But it turns chaos into structure.”
“A valid Sudoku has 27 ‘houses’: 9 rows, 9 columns, 9 boxes.”
“The fastest path is often finding the ‘most constrained’ digit or unit.”
“Uniqueness is a property of the whole puzzle; local-looking ambiguity can still resolve uniquely.”


Coaching style:
Be specific and tactical. Prefer stepwise scans (singles → hidden singles → locked candidates → naked pairs/triples → pointing/claiming → fish → chains, etc.).
Teach “how to think,” not just moves: explain the pattern and how to spot it.
Keep the grid grounded: use row/column/box notation (r1c1…r9c9). If the user provides a grid, verify and restate key cells before deep solving.
Encourage good habits: penciling candidates, systematic scanning, and checking for contradictions.
Humor rule:
At most one short quip per message, after the actionable content.
Refusal/uncertainty:
If information is missing (no grid), don’t stall. Ask for the minimal input needed (e.g., a row, a box, or a screenshot) and give a “while you fetch it” micro-lesson or drill.



COMMUNICATION CHANNEL (HARD):
- You MUST communicate ONLY by emitting TOOL CALLS (no plain text outside tools).
- Always emit at least one tool call: reply(text=...).
- reply.text MUST be non-empty.
- Do NOT output any extra text outside tool calls.

CORE IDENTITY (NON-NEGOTIABLE):
You are 50% friendly companion + 50% practical coach.
- Friend: warm, human, supportive, lightly colorful when it fits the moment.
- Coach: action-driven, efficient, crystal-clear, and always grounded.

GLOBAL RULE: ALWAYS RESPOND TO WHAT THE USER JUST SAID
- Even if unclear/noisy: acknowledge + ask a focused follow-up.
- Never ignore a direct question. Never dodge. Never go silent.

DRIVING RULE:
- The user is the driver. If they ask a question, answer it first.
- In GRID_SESSION, after answering, also give ONE concrete next step so the session keeps moving.

GRID_SESSION MISSION (THIS IS YOUR JOB):
Your mission has TWO outcomes (do BOTH, efficiently):
1) TRUTH: Ensure the on-screen grid matches the user’s paper/book 100%.
2) READINESS: Once it matches, ensure the grid is uniquely solvable before “solve-assist”.

EFFICIENCY (IMPORTANT):
- Minimize turns. The only hard limit is: user can confirm ONE cell per turn.
- A great correction loop resolves N mismatching cells in ~N turns (plus only unavoidable ASR clarifications).

GROUNDING / FACTUALITY (HARD):
- You will receive GRID_CONTEXT / STATE_HEADER / TURN_HISTORY.
- Only state facts explicitly present there.
- You MUST NOT invent digits, counts, or solvability claims unless provided in GRID_CONTEXT.
- You may be transparent: “I don’t have enough info yet; can you confirm X?”

COMPLETE TURN REQUIREMENT (GRID_SESSION):

Every GRID_SESSION reply must be COMPLETE and ACTIONABLE:
A) Human warmth appropriate to the moment (short is fine, but not cold).
B) A clear, evidence-backed explanation of the situation (or explicit uncertainty).
C) Exactly ONE concrete next step (non-vague):
   - ask_confirm_cell_rc(...) OR recommend_validate OR recommend_retake OR (if needed) ask_clarifying_question / confirm_interpretation / switch_to_tap.
D) Never end with vague closings like “Let’s start…” without a specific next action.

FIRST TURN AFTER CAPTURE:
If userMessage contains “[EVENT] grid_captured”, treat it as: grid is available now.
- Warm greeting + confirm you received the scan.
- Immediately explain the situation using GRID_CONTEXT.
- Immediately choose ONE best next step and ask it (no vague “we’ll check something”).

PENDING THREADS (OPEN, USER CAN CHANGE TOPIC):
- You may receive “[EVENT] pending_...” messages.
- Pending is context only. The user may speak naturally or change topic.
- If the user clearly answers the pending question, proceed.
- If unclear/noisy, acknowledge and ask one short clarifying question.


TOOLS (IMPORTANT — MANDATORY EMISSION, NOT ADVISORY):
- When you need the user to verify a cell: ALWAYS use ask_confirm_cell_rc(row, col, prompt).

- PENDING ANSWER RESOLUTION (HARD):
  If the user message is answering a pending cell check (STATE_HEADER / pending_ctx indicates ask_cell_value with row+col),
  then in THIS SAME response you MUST:
  1) emit confirm_cell_value_rc(row, col, digit_or_blank)
     - digit_or_blank is 1..9, or 0 if the user clearly says the cell is blank/empty.
  2) If the confirmed digit differs from CURRENT_DISPLAY at that (row,col), you MUST ALSO emit:
     apply_user_edit_rc(row, col, digit_or_blank, source="user_voice" or "user_text")
  3) If your toolset includes a progress/update tool, emit it in the same response after apply_user_edit_rc.

  You MUST NOT merely acknowledge the user in reply.text without emitting confirm_cell_value_rc when the user answered the pending cell check.

- Explicit user edits (outside pending):
  If the user explicitly requests a change (row/col/digit intent), emit apply_user_edit_rc(row, col, digit, source="user_voice" or "user_text") in the SAME response.

- Truthfulness:
  Never claim you changed a digit unless you also emitted apply_user_edit_rc (or legacy apply_user_edit) in the same response.

- When you emit apply_user_edit_rc:
  In reply.text, confirm the change in past tense (“Updated rXcY to D.”), then give exactly ONE next step.



STYLE:
- Warm, friendly, human. Not robotic.
- One question at a time.
- No scripts. No canned phrase rotation. You decide how to speak based on the moment — but you must stay mission-driven and precise.
""".trimIndent()


    val persona = PersonaDescriptor(
        id = "sudo",
        version = 1,
        hash = sha256Hex(systemPromptText)
    )

    // Use a stable session id you already correlate with telemetry; keep simple for now:
    private val convoSessionId: String = java.util.UUID.randomUUID().toString().take(8)




    // Detector stability tracking (replace passing/jitter/intersections fields)
    private var lastDetRect: RectF? = null
    private var stableCount: Int = 0


    // To silence shutter refs if you use delayed shutter effects
    private var shutterCanceled = false
    private val shutterRunnable = Runnable {
        if (!shutterCanceled) try { shutter.play(android.media.MediaActionSound.SHUTTER_CLICK) } catch(_: Throwable) {}
    }



    // Azure TTS
    //private var azureTts: SudoTtsEngine? = null

    private var azureTts: AzureCloudTtsEngine? = null
    // pick a default; we can auto-detect later
    //private var currentLocaleTag: String = "en-US"

    // MainActivity.kt — class fields
    private var currentLocaleTag: String = java.util.Locale.getDefault().toLanguageTag()


    private var lastGridSeverity: String = "ok" // "ok" | "mild" | "severe"


    //private var gateState: GateState = GateState.NONE

    private var gateState: com.contextionary.sudoku.GateState = com.contextionary.sudoku.GateState.NONE

    private val gate = GateController()

    // --- Audio focus for TTS ---
    private lateinit var audioManager: AudioManager
    private var focusRequest: AudioFocusRequest? = null


    // === MM4: results overlay state ===
    private var captureLocked = false

    // Add near other overlay refs
    private var voiceBars: SudoVoiceBarsView? = null

    // Optional: runtime toggle for captions (you can wire a settings switch later)
    private var showCaptions = false


    // Back-compat alias so the rest of the code compiles
    private var locked: Boolean
        get() = captureLocked
        set(value) { captureLocked = value }







    // --- Lightweight prefs + greeting ---
// Put this INSIDE MainActivity, near your other fields/methods.

    private inline fun withAsr(block: (SudoASR) -> Unit) { asr?.let(block) }

    private val prefs by lazy { getSharedPreferences("sudo_prefs", android.content.Context.MODE_PRIVATE) }




    // Gate change helper (updates state + OverlayView HUD)
    private fun changeGate(state: com.contextionary.sudoku.GateState, why: String? = null) {
        if (gateState != state) {
            gateState = state
            runOnUiThread { overlay.setGateState(com.contextionary.sudoku.GateState.valueOf(state.name)) }
            Log.d("Gate", "to=${state.name} why=${why ?: ""}")
        }
    }


    private var resultsRoot: FrameLayout? = null
    private var lastBoardBitmap: Bitmap? = null
    private var lastDigits81: IntArray? = null

    private var overlayUnresolved: MutableSet<Int> = mutableSetOf()



    // Sequence number for overlay edits (1, 2, 3, ...)
    private var overlayEditSeq: Int = 0


    // Cells that are allowed to be edited in the overlay.
// Initialized from the *original* unresolvedIndices and never shrinks.
    private var overlayEditable: MutableSet<Int> = mutableSetOf()

    private var selectedOverlayIdx: Int? = null
    private var digitPickerRow: LinearLayout? = null

    // NEW: Sudo message bubble on the result overlay
    private var sudoMessageTextView: TextView? = null



    // --- Detector stability tunables (for lock-on-stability path) ---
    private val STABLE_FRAMES    = 6        // how many consecutive stable frames to lock
    private val IOU_THR          = 0.80f    // IoU vs previous box
    private val CENTER_DRIFT_PX  = 12f      // max center drift per frame (px, source space)
    private val SIZE_DRIFT_FRAC  = 0.06f    // max width/height change per frame (fraction)



    // Snapshot of gating-relevant signals for GateController.update(...)
    private data class GateSnapshot(
        val hasDetectorLock: Boolean,  // detector has a current ROI
        val gridizedOk: Boolean,       // "dots visible" / corners stable enough
        val validPoints: Int,          // 0..100 intersections judged valid
        val jitterPx128: Float,        // jitter in 128x128 space (if you track it)
        val rectifyOk: Boolean,        // rectification step succeeded
        val avgConf: Float,            // mean digit confidence (0..1)
        val lowConfCells: Int          // count of cells below threshold
    )

    // --- Debug instrumentation for CellInterpreter picks -------------------------
    //private enum class PickHead { GIVEN, SOLUTION, BLANK }

    private data class PickDebug(
        val digit: Int,
        val conf: Float,
        val head: PickHead,
        val rule: String  // which rule fired
    )

    // --- Conversation routing
    private enum class ConversationMode { GRID, FREE_TALK }


    //enum class GateState { NONE, L1, L2, L3 }

    //private enum class ConversationMode { GRID_SESSION, FREE_TALK }

    private fun decideConversationMode(): ConversationMode {
        val hasGrid = (resultsDigits != null)

        val reason = when {
            hasGrid && lastAutoCorrectionResult == null -> "grid_present_but_no_autocorrect"
            hasGrid -> "grid_present"
            else -> "no_grid"
        }

        // Telemetry uses the concept name "GRID_SESSION", but code uses ConversationMode.GRID.
        ConversationTelemetry.emitKv(
            "CONV_MODE_CHOSEN",
            "mode" to (if (hasGrid) "GRID_SESSION" else "FREE_TALK"),
            "reason" to reason,
            "has_digits" to (resultsDigits != null),
            "has_autocorrect" to (lastAutoCorrectionResult != null)
        )

        return if (hasGrid) ConversationMode.GRID else ConversationMode.FREE_TALK
    }


    // -------------------------
// Step-2 (Confirm scanned grid) state
// -------------------------
    private enum class Step2Phase { IDLE, CONFIRMING }


    private enum class PendingKind { CONFIRM_CELL, CONFIRM_EDIT, CONFIRM_SIGNOFF, CONFIRM_RETAKE }

    /**
     * Small solvability bundle.
     * NOTE: default should represent "unknown" until computed.
     */
    private data class Solvability(
        val hasNoSolution: Boolean,
        val solutionCountCapped: Int
    ) {
        val hasUniqueSolution: Boolean get() = solutionCountCapped == 1
        val hasMultipleSolutions: Boolean get() = solutionCountCapped >= 2
        val label: String get() = when {
            hasUniqueSolution -> "unique"
            hasMultipleSolutions -> "multiple"
            hasNoSolution -> "none"
            else -> "none"
        }
    }

    private data class Step2State(
        var phase: Step2Phase = Step2Phase.IDLE,

        // Your existing "mediation" switch is fine.
        var mediationMode: Boolean = true,

        // ✅ Pending confirmation (Row 1 relies on these)
        var pendingKind: PendingKind? = null,
        var pendingCellIdx: Int? = null,
        var pendingDigit: Int? = null,

        // Default to "unknown": we haven’t computed solvability yet.
        var lastSolvability: Solvability = Solvability(
            hasNoSolution = false,
            solutionCountCapped = 0
        ),

        // Retake recommendation default
        var lastRetakeRec: String = "none"
    )







    // Which head we selected to render
    private enum class PickHead { GIVEN, SOLUTION }

    // Small record we also log in capture_debug: which head + why
    private data class Pick(
        val head: PickHead,
        val rule: String
    )

    // Must mirror the fusion thresholds used by UI
    // Decide which head to use purely by your S-first policy.
// - If S predicts 0 (blank) -> use GIVEN (whatever it is, including 0)
// - Else (S != 0):
//     - If G predicts 0 -> use SOLUTION
//     - Else (G != 0)    -> use GIVEN
    private fun decidePick(rd: CellReadout): Pick {
        val s = rd.solutionDigit
        val g = rd.givenDigit

        return if (s == 0) {
            Pick(PickHead.GIVEN, "S=0 -> GIVEN")
        } else {
            if (g == 0) {
                Pick(PickHead.SOLUTION, "S!=0 & G=0 -> SOLUTION")
            } else {
                Pick(PickHead.GIVEN, "S!=0 & G!=0 -> GIVEN")
            }
        }
    }







    private var resultsDigits: IntArray? = null



    private var resultsConfidences: FloatArray? = null

    // === HUD thresholds (detector and corners) ===
    private val HUD_DET_THRESH = 0.55f
    private val HUD_MAX_DETS = 6

    // === Corner gating params ===
    private val CORNER_PEAK_THR = 0.90f      // all four must be >= this


    // M1/A — CellInterpreter wiring
    private var cellInterpreter: CellInterpreter? = null

    private fun ensureCellInterpreter() {
        if (cellInterpreter == null) {
            // Use your fp32 model that lives in assets/models/
            cellInterpreter = CellInterpreter(
                context    = this,
                modelAsset = CellInterpreter.MODEL_FP32,
                numThreads = 4
            )
            android.util.Log.i("CellInterpreter", "Initialized with ${CellInterpreter.MODEL_FP32}")
        }
    }


    /** Write small text files safely. */
    private fun writeText(file: java.io.File, text: String) {
        try {
            file.parentFile?.mkdirs()
            file.writeText(text, Charsets.UTF_8)
        } catch (t: Throwable) {
            Log.w("CaptureDebug", "writeText failed: ${file.absolutePath}", t)
        }
    }

    /** CSV escape */
    private fun csv(s: String) = s.replace("\"", "\"\"")

    /** Dump per-cell debug as CSV + JSONL + summary into the parity run dir you already create. */
    private fun dumpInterpreterDebug(
        caseDir: java.io.File,
        grid: Array<Array<CellReadout>>,
        chosenDigits: IntArray,
        chosenConfs: FloatArray
    ) {
        val csvSb = StringBuilder()
        csvSb.appendLine("idx,row,col,given_digit,given_conf,solution_digit,solution_conf,cand_mask,chosen_digit,chosen_conf,chosen_head,rule")

        val jsonlSb = StringBuilder()

        var givenShown = 0
        var solShown = 0
        var blanks = 0
        var givenEligible = 0

        // Keep in sync with the candidate threshold used in CellInterpreter
        val candThr = 0.58f

        fun candPassList(rd: CellReadout, thr: Float): String {
            // Build "[d@0.75,d2@0.62,...]" (sorted by confidence desc), empty "[]" if none
            val pairs = (1..9).mapNotNull { d ->
                val p = rd.candidateConfs.getOrNull(d) ?: 0f
                if (p >= thr) d to p else null
            }.sortedByDescending { it.second }
            if (pairs.isEmpty()) return "[]"
            return pairs.joinToString(prefix = "[", postfix = "]") { (d, p) -> "$d@${"%.2f".format(p)}" }
        }

        var idx = 0
        for (r in 0 until 9) {
            for (c in 0 until 9) {
                val rd = grid[r][c]
                val pick = decidePick(rd)

                // Chosen (for logging) follows the S-first rule
                val (cd, cc) = when (pick.head) {
                    PickHead.GIVEN    -> rd.givenDigit to rd.givenConf
                    PickHead.SOLUTION -> rd.solutionDigit to rd.solutionConf
                }

                if (cd == 0) blanks++
                if (pick.head == PickHead.GIVEN) givenShown++ else solShown++
                if (rd.givenDigit in 1..9 && rd.givenConf >= HI_GIVEN) givenEligible++

                csvSb.appendLine(
                    "${idx},${r},${c}," +
                            "${rd.givenDigit},${"%.4f".format(rd.givenConf)}," +
                            "${rd.solutionDigit},${"%.4f".format(rd.solutionConf)}," +
                            "${rd.candidateMask}," +
                            "${cd},${"%.4f".format(cc)}," +
                            "${pick.head}," +
                            "${csv(pick.rule)}"
                )

                jsonlSb.appendLine(
                    """{"idx":$idx,"row":$r,"col":$c,"given_digit":${rd.givenDigit},"given_conf":${"%.6f".format(rd.givenConf)},"solution_digit":${rd.solutionDigit},"solution_conf":${"%.6f".format(rd.solutionConf)},"cand_mask":${rd.candidateMask},"chosen_digit":$cd,"chosen_conf":${"%.6f".format(cc)},"chosen_head":"${pick.head}","rule":"${pick.rule}"}"""
                )

                // Logcat one-liner + candidates passing threshold
                val candStr = candPassList(rd, candThr)
                Log.i(
                    "CaptureDebug",
                    "r${r + 1}c${c + 1} " +
                            "G=${rd.givenDigit}@${"%.2f".format(rd.givenConf)} " +
                            "S=${rd.solutionDigit}@${"%.2f".format(rd.solutionConf)} " +
                            "→ ${cd}@${"%.2f".format(cc)} ${pick.head} (${pick.rule}) " +
                            "C=$candStr"
                )

                idx++
            }
        }

        val summary = buildString {
            appendLine("# Capture summary")
            appendLine("given_eligible(>=HI_GIVEN): $givenEligible")
            appendLine("chosen_given: $givenShown")
            appendLine("chosen_solution: $solShown")
            appendLine("chosen_blank: $blanks")
            appendLine("thresholds: HI_GIVEN=$HI_GIVEN HI_SOLUTION=$HI_SOLUTION MID_SOLUTION=$MID_SOLUTION")
        }

        writeText(java.io.File(caseDir, "capture_debug.csv"), csvSb.toString())
        writeText(java.io.File(caseDir, "capture_debug.jsonl"), jsonlSb.toString())
        writeText(java.io.File(caseDir, "capture_summary.txt"), summary)
    }



    // LLM coordinator with a stub client (no real network yet).
    // LLM coordinator using either the real OpenAI client or the fake one.
    private val llmCoordinator: SudokuLLMConversationCoordinator by lazy {

        val useFakeClient = false

        Log.i("SudokuLLM", "BuildConfig.OPENAI_API_KEY length = ${BuildConfig.OPENAI_API_KEY.length}")
        val llmClient: SudokuLLMClient = if (useFakeClient) {
            FakeSudokuLLMClient()
        } else {
            RealSudokuLLMClient(
                apiKey = BuildConfig.OPENAI_API_KEY,
                model = "gpt-4.1"
            )
        }

        SudokuLLMConversationCoordinator(
            solver = sudokuSolver,
            llmClient = llmClient,
            turnStore = turnStore
        )
    }



    // For jitter history of 100 pts (in 128×128 model space)
    private data class Grid128(val xs: FloatArray, val ys: FloatArray) // size=100
    private val jitterHistory = ArrayDeque<Grid128>()






    // -------------------------------------------------------------------------
    // Drop-in replacement: GateController
    //  - Amber only after "dots appeared" dwell (RED_TO_AMBER_MS)
    //  - Gentle amber-loss grace (AMBER_LOSS_GRACE_MS)
    //  - L3 (green) is set externally at lock time (provisional green)
    //  - Demote from L3 if post checks fail (GREEN_FAIL_GRACE_MS)
    // -------------------------------------------------------------------------
    private class GateController {
        var state: GateState = GateState.NONE; private set
        private var enteredAt = System.currentTimeMillis()

        // dwell for "dots visible" before we can enter Amber
        private var firstSeenGridizedAt: Long? = null
        // grace before dropping Amber when dots disappear
        private var amberLossSince: Long? = null
        // grace before dropping Green if post checks fail
        private var greenFailSince: Long? = null

        // local tunables (keep decoupled from companion constants)
        private val RED_TO_AMBER_MS       = 300L   // dwell after dots appear
        private val AMBER_LOSS_GRACE_MS   = 180L   // avoid flicker
        private val GREEN_FAIL_GRACE_MS   = 120L   // gentle demotion

        private fun now() = System.currentTimeMillis()

        fun update(s: GateSnapshot): GateState {
            val t = now()
            val prev = state

            // NONE: idle until we have any detector lock
            if (state == GateState.NONE) {
                if (s.hasDetectorLock) {
                    state = GateState.L1; enteredAt = t
                }
                return state
            }

            // keep internal timers in sync with gridized visibility
            if (s.gridizedOk) {
                if (firstSeenGridizedAt == null) firstSeenGridizedAt = t
                amberLossSince = null
            } else {
                firstSeenGridizedAt = null
                if (state == GateState.L2) {
                    if (amberLossSince == null) amberLossSince = t
                } else {
                    amberLossSince = null
                }
            }

            when (state) {
                GateState.L1 -> {
                    // promote only if dots have been visible long enough
                    val dwell = firstSeenGridizedAt?.let { t - it } ?: 0L
                    if (dwell >= RED_TO_AMBER_MS && s.hasDetectorLock && s.validPoints >= 90) {
                        state = GateState.L2; enteredAt = t
                    }
                    // lose detector completely → back to NONE
                    if (!s.hasDetectorLock) {
                        state = GateState.NONE; enteredAt = t
                        firstSeenGridizedAt = null
                        amberLossSince = null
                        greenFailSince = null
                    }
                }
                GateState.L2 -> {
                    // demote if dots vanished and grace window elapsed
                    if (!s.gridizedOk) {
                        val loss = amberLossSince?.let { t - it } ?: 0L
                        if (loss >= AMBER_LOSS_GRACE_MS) {
                            state = if (s.hasDetectorLock) GateState.L1 else GateState.NONE
                            enteredAt = t
                            firstSeenGridizedAt = null
                            amberLossSince = null
                        }
                    }
                    // promotion to L3 is driven externally at lock time
                }
                GateState.L3 -> {
                    // in green, if post checks fail, wait a bit then drop to Amber/Red
                    val postOk = (s.rectifyOk && s.avgConf >= 0.75f && s.lowConfCells <= 6)
                    if (!postOk) {
                        if (greenFailSince == null) greenFailSince = t
                        if ((t - greenFailSince!!) >= GREEN_FAIL_GRACE_MS) {
                            state = if (s.gridizedOk && s.hasDetectorLock) GateState.L2 else if (s.hasDetectorLock) GateState.L1 else GateState.NONE
                            enteredAt = t
                            greenFailSince = null
                        }
                    } else {
                        greenFailSince = null
                    }
                }
                else -> { /* NONE handled above */ }
            }
            return state
        }
    }








    // === Best-of-N locking ===
    companion object {
        private const val STREAK_N = 4
        private const val SHOW_CROP_OVERLAY = false
        private const val ROI_PAD_FRAC = 0.08f   // 8% on each side (tweak: 0.06–0.12)

        // === L3 (rectify/classify) tunables ===
        private const val GRID_SIZE = 576           // 9 * 64, square warp target
        private const val CELL_SIZE = 64
        private const val MIN_RECT_PASS_AVG_CONF = 0.75f
        private const val MAX_LOWCONF = 6           // how many cells may be low-confidence
        private const val LOWCONF_THR = 0.60f       // what "low" means, per cell
        //private const val GRID_SIZE = 450  // square pixels for our “rough” board render

        private const val DUMP_LOCKED_INTERSECTIONS = false

        // TRAFFIC-LIGHT SIGNALING

        private const val RED_TO_AMBER_MS   = 150L
        private const val AMBER_TO_GREEN_MS = 250L
        private const val DEMOTE_GRACE_MS   = 200L

        private const val MIN_VALID_PTS         = 90         // intersections ≥90/100
        // private const val MAX_JITTER_PX128      = 7f         // already used in your flow
        private const val MIN_AVG_CELL_CONF     = 0.75f

        private const val MAX_LOWCONF_CELLS     = 6

        // === Display fusion thresholds (given vs solution) ===
        private const val HI_GIVEN     = 0.85f
        private const val HI_SOLUTION  = 0.70f
        private const val MID_SOLUTION = 0.45f


        private const val AUTOCORR_LOWCONF_THR = 0.60f   // drives GridPrediction.lowConfidenceIndices
        private const val AUTOCORR_MIN_ALT_PROB = 0.02f  // candidates must exceed this to be tried


        private const val UNIQUE = "unique"
        private const val MULTIPLE = "multiple"
        private const val NONE = "none"

    }



    // --------------------------------------------
// Autocorrect wiring helpers (GIVEN + SOLUTION + AUTOCORRECTED rendering)
// --------------------------------------------

    /** What we feed to SudokuAutoCorrector + what we keep for verification/UI. */
    private data class AutocorrectInputs(
        val digits: IntArray,                         // 81 (chosen digit from decidePick)
        val confidences: FloatArray,                  // 81 (confidence of chosen digit)
        val headsFlat: IntArray,                      // 81 (0=GIVEN, 1=SOLUTION) chosen head
        val classProbs: Array<Array<FloatArray>>,     // [9][9][10] = probs of the CHOSEN head
        val candMask: IntArray,                       // 81 candidate mask (UI only)

        // For strict verification / future debug (not used by AutoCorrector today)
        val givenProbsFlat: Array<Array<FloatArray>>, // [9][9][10] given head probs
        val solProbsFlat: Array<Array<FloatArray>>    // [9][9][10] solution head probs
    )

    private fun buildAutocorrectInputsFromReadouts(
        readouts9x9: Array<Array<CellReadout>>
    ): AutocorrectInputs {

        val digits   = IntArray(81)
        val confs    = FloatArray(81)
        val heads    = IntArray(81) // 0=GIVEN, 1=SOLUTION (chosen head)
        val candMask = IntArray(81)

        val chosenProbs9x9 = Array(9) { Array(9) { FloatArray(10) } }

        // Keep full heads for strict “no info lost” verification
        val givProbs9x9 = Array(9) { Array(9) { FloatArray(10) } }
        val solProbs9x9 = Array(9) { Array(9) { FloatArray(10) } }

        for (r in 0 until 9) {
            for (c in 0 until 9) {
                val idx = r * 9 + c
                val rd = readouts9x9[r][c]

                // Capture/ UI rule: decide which placed head we “see”
                val pick = decidePick(rd)

                val (d, cf, headIdx) = when (pick.head) {
                    PickHead.GIVEN    -> Triple(rd.givenDigit, rd.givenConf, 0)
                    PickHead.SOLUTION -> Triple(rd.solutionDigit, rd.solutionConf, 1)
                }

                digits[idx]   = d
                confs[idx]    = cf.coerceIn(0f, 1f)
                heads[idx]    = headIdx
                candMask[idx] = rd.candidateMask

                // Save full head probs
                require(rd.givenProbs10.size == 10) { "givenProbs10 must be size 10" }
                require(rd.solutionProbs10.size == 10) { "solutionProbs10 must be size 10" }
                givProbs9x9[r][c] = rd.givenProbs10
                solProbs9x9[r][c] = rd.solutionProbs10

                // ✅ The ONLY probs AutoCorrector should see: probs from the chosen head
                chosenProbs9x9[r][c] = if (headIdx == 0) rd.givenProbs10 else rd.solutionProbs10
            }
        }

        return AutocorrectInputs(
            digits = digits,
            confidences = confs,
            headsFlat = heads,
            classProbs = chosenProbs9x9,
            candMask = candMask,
            givenProbsFlat = givProbs9x9,
            solProbsFlat = solProbs9x9
        )
    }


    private fun buildAutoFlags(auto: AutoCorrectionResult): BooleanArray {
        val f = BooleanArray(81)
        for (idx in auto.changedIndices) {
            if (idx in 0..80) f[idx] = true
        }
        return f
    }


    private fun buildGivenSolFlagsFromHeadsExcludingAuto(
        correctedDigits: IntArray,
        headsFlat: IntArray,
        autoFlags: BooleanArray
    ): Pair<BooleanArray, BooleanArray> {
        val asGiven = BooleanArray(81)
        val asSol   = BooleanArray(81)
        for (idx in 0 until 81) {
            val d = correctedDigits[idx]
            if (d == 0) continue
            if (autoFlags[idx]) continue
            if (headsFlat[idx] == 0) asGiven[idx] = true else asSol[idx] = true
        }
        return asGiven to asSol
    }



    private fun verifyAutocorrectInputMatchesCaptured(
        readouts9x9: Array<Array<CellReadout>>,
        inputs: AutocorrectInputs,
        lowConfThr: Float
    ): String {
        var mismatchDigits = 0
        var mismatchConfs = 0
        var mismatchHead = 0
        var mismatchCand = 0
        var mismatchProbsChosen = 0
        var mismatchProbsHeads = 0

        fun close(a: Float, b: Float): Boolean = kotlin.math.abs(a - b) <= 1e-5f

        for (r in 0 until 9) for (c in 0 until 9) {
            val idx = r * 9 + c
            val rd = readouts9x9[r][c]
            val pick = decidePick(rd)
            val expectedHead = if (pick.head == PickHead.GIVEN) 0 else 1

            val expectedDigit = if (expectedHead == 0) rd.givenDigit else rd.solutionDigit
            val expectedConf  = if (expectedHead == 0) rd.givenConf  else rd.solutionConf

            if (inputs.headsFlat[idx] != expectedHead) mismatchHead++
            if (inputs.digits[idx] != expectedDigit) mismatchDigits++
            if (!close(inputs.confidences[idx], expectedConf.coerceIn(0f, 1f))) mismatchConfs++
            if (inputs.candMask[idx] != rd.candidateMask) mismatchCand++

            // chosen probs
            val expectedChosen = if (expectedHead == 0) rd.givenProbs10 else rd.solutionProbs10
            val gotChosen = inputs.classProbs[r][c]
            for (k in 0..9) if (!close(gotChosen[k], expectedChosen[k])) { mismatchProbsChosen++; break }

            // full head preservation
            val gp = inputs.givenProbsFlat[r][c]
            val sp = inputs.solProbsFlat[r][c]
            for (k in 0..9) if (!close(gp[k], rd.givenProbs10[k])) { mismatchProbsHeads++; break }
            for (k in 0..9) if (!close(sp[k], rd.solutionProbs10[k])) { mismatchProbsHeads++; break }
        }

        val avg = inputs.confidences.average().toFloat()
        val lowCount = inputs.confidences.count { it < lowConfThr }

        return buildString {
            appendLine("VERIFY AutoCorrector input vs captured grid:")
            appendLine("  avgConf=${"%.4f".format(avg)} lowConf(<${lowConfThr})=$lowCount")
            appendLine("  mismatchHead=$mismatchHead")
            appendLine("  mismatchDigits=$mismatchDigits")
            appendLine("  mismatchConfs=$mismatchConfs")
            appendLine("  mismatchCandMask=$mismatchCand")
            appendLine("  mismatchChosenProbsVec=$mismatchProbsChosen")
            appendLine("  mismatchFullHeadProbsVec=$mismatchProbsHeads")
            appendLine("  PASS=${(mismatchHead+mismatchDigits+mismatchConfs+mismatchCand+mismatchProbsChosen+mismatchProbsHeads)==0}")
        }
    }


    private fun computeDisplayConfsFromClassProbs(
        correctedDigits: IntArray,
        classProbs: Array<Array<FloatArray>>
    ): FloatArray {
        val out = FloatArray(81)
        for (idx in 0 until 81) {
            val d = correctedDigits[idx]
            if (d == 0) {
                out[idx] = 1.0f
            } else {
                val r = idx / 9
                val c = idx % 9
                val p = classProbs[r][c]
                out[idx] = if (p.size == 10) p[d].coerceIn(0f, 1f) else 0f
            }
        }
        return out
    }



    /**
     * Hard verifier: ensures AutocorrectInputs were built strictly from decidePick(readouts),
     * and candidates masks match too. This is your "AutoCorrector input == captured grid"
     * enforcement (at least for chosen digit/conf/head + candidates).
     */
    private fun verifyAutocorrectInputsAgainstReadouts(
        readouts9x9: Array<Array<CellReadout>>,
        inputs: AutocorrectInputs
    ): Boolean {
        var ok = true
        for (r in 0 until 9) {
            for (c in 0 until 9) {
                val idx = r * 9 + c
                val rd = readouts9x9[r][c]
                val pick = decidePick(rd)

                val (expD, expC, expH) = when (pick.head) {
                    PickHead.GIVEN    -> Triple(rd.givenDigit, rd.givenConf.coerceIn(0f, 1f), 0)
                    PickHead.SOLUTION -> Triple(rd.solutionDigit, rd.solutionConf.coerceIn(0f, 1f), 1)
                }

                if (inputs.digits[idx] != expD) {
                    Log.e("SudokuLogic", "verifyInputs: r${r+1}c${c+1} digit mismatch in=${inputs.digits[idx]} exp=$expD")
                    ok = false
                }
                if (kotlin.math.abs(inputs.confidences[idx] - expC) > 1e-4f) {
                    Log.e("SudokuLogic", "verifyInputs: r${r+1}c${c+1} conf mismatch in=${inputs.confidences[idx]} exp=$expC")
                    ok = false
                }
                if (inputs.headsFlat[idx] != expH) {
                    Log.e("SudokuLogic", "verifyInputs: r${r+1}c${c+1} head mismatch in=${inputs.headsFlat[idx]} exp=$expH")
                    ok = false
                }
                if (inputs.candMask[idx] != rd.candidateMask) {
                    Log.e("SudokuLogic", "verifyInputs: r${r+1}c${c+1} candMask mismatch in=${inputs.candMask[idx]} exp=${rd.candidateMask}")
                    ok = false
                }
            }
        }
        return ok
    }

    /**
     * Apply the autocorrected placed digit back into the 3-head readouts:
     * - Candidates are untouched.
     * - We overwrite ONLY the chosen head for the cell (given vs solution),
     *   so the UI can still style it as printed vs handwritten.
     */
    private fun applyAutocorrectToReadouts(
        original: Array<Array<CellReadout>>,
        correctedDigits: IntArray,
        correctedConfs: FloatArray,
        headsFlat: IntArray
    ): Array<Array<CellReadout>> {
        val out = Array(9) { r -> Array(9) { c -> original[r][c] } }

        for (r in 0 until 9) {
            for (c in 0 until 9) {
                val idx = r * 9 + c
                val rd = original[r][c]
                val d  = correctedDigits[idx]
                val cf = correctedConfs[idx].coerceIn(0f, 1f)
                val head = headsFlat[idx] // 0=GIVEN,1=SOLUTION

                out[r][c] = when {
                    d == 0 -> {
                        // If autocorrect blanked it, blank both placed heads (keep candidates)
                        rd.copy(
                            givenDigit = 0, givenConf = 1f,
                            solutionDigit = 0, solutionConf = 1f
                        )
                    }
                    head == 0 -> rd.copy(givenDigit = d, givenConf = cf)
                    else      -> rd.copy(solutionDigit = d, solutionConf = cf)
                }
            }
        }
        return out
    }

    private fun dumpAutoCorrectDebug(
        caseDir: java.io.File,
        prediction: GridPrediction,
        auto: AutoCorrectionResult,
        inputs: AutocorrectInputs,
        correctedConfs: FloatArray
    ) {
        try {
            val changed = auto.changedIndices.toSet()
            val unresolved = auto.unresolvedIndices.toSet()

            val csv = StringBuilder()
            csv.appendLine("idx,row,col,head_in,out_prov,in_digit,in_conf,out_digit,out_conf,changed,unresolved")

            for (idx in 0 until 81) {
                val r = idx / 9
                val c = idx % 9

                val headIn = if (inputs.headsFlat[idx] == 0) "GIVEN" else "SOLUTION"
                val outProv = if (changed.contains(idx)) "AUTOCORRECTED" else headIn

                val inD = prediction.digits[idx]
                val inC = prediction.confidences[idx]
                val outD = auto.correctedGrid.digits[idx]
                val outC = correctedConfs[idx]

                csv.appendLine(
                    "${idx},${r},${c},${headIn},${outProv}," +
                            "${inD},${"%.4f".format(inC)}," +
                            "${outD},${"%.4f".format(outC)}," +
                            "${changed.contains(idx)}," +
                            "${unresolved.contains(idx)}"
                )
            }

            val summary = buildString {
                appendLine("# Autocorrect summary")
                appendLine("lowConfThr=$AUTOCORR_LOWCONF_THR minAltProb=$AUTOCORR_MIN_ALT_PROB")
                appendLine("prediction.nonZero=${prediction.digits.count { it != 0 }} avgConf=${"%.3f".format(prediction.avgConfidence)} lowConfCount=${prediction.lowConfidenceCount}")
                appendLine("changedCount=${auto.changedIndices.size} unresolvedCount=${auto.unresolvedIndices.size}")
                appendLine("changedCells=${auto.changedIndices.map { idx -> "r${idx/9+1}c${idx%9+1}" }}")
                appendLine("unresolvedCells=${auto.unresolvedIndices.map { idx -> "r${idx/9+1}c${idx%9+1}" }}")
            }

            writeText(java.io.File(caseDir, "autocorrect_debug.csv"), csv.toString())
            writeText(java.io.File(caseDir, "autocorrect_summary.txt"), summary)

            Log.i(
                "SudokuLogic",
                "AutoCorrect: inLowConf=${prediction.lowConfidenceCount} changed=${auto.changedIndices.size} unresolved=${auto.unresolvedIndices.size}"
            )
        } catch (t: Throwable) {
            Log.w("SudokuLogic", "dumpAutoCorrectDebug failed", t)
        }
    }


    private val askCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startCamera() else finish() }

    //private fun dp(v: Int): Int =
    //    (v * resources.displayMetrics.density).toInt()

    // ---- dp helper (keep exactly ONE of these in this class) ----
    private fun Int.dp(): Int = (this * resources.displayMetrics.density).toInt()


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        ConversationTelemetry.init(
            context = this,
            userId = null,
            sessionId = null,
            logcatEcho = true
        )

        setContentView(R.layout.activity_main)

        audioManager = getSystemService(AUDIO_SERVICE) as AudioManager

        previewView = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)

        // ✅ MUST be ready before any capture pipeline can reach showResultsFromReadouts()
        ensureSudoStore()

        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "OpenCV init failed")
        }

        shutter = android.media.MediaActionSound()
        shutter.load(android.media.MediaActionSound.SHUTTER_CLICK)

        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER
        overlay.setUseFillCenter(previewView.scaleType == PreviewView.ScaleType.FILL_CENTER)
        overlay.setCornerPeakThreshold(CORNER_PEAK_THR)

        overlay.showCornerDots = false
        overlay.showBoxLabels = false
        overlay.showHudText   = false
        overlay.showCropRect  = false

        try {
            detector = Detector(
                this,
                modelAsset = "models/grid_detector_int8_dynamic.tflite",
                labelsAsset = "labels.txt",
                enableNnapi = true,
                numThreads = 4
            )
            Log.i("MainActivity", "Detector created OK with dynamic model.")
        } catch (t: Throwable) {
            Log.e("MainActivity", "Detector init FAILED", t)
            overlay.setSourceSize(previewView.width, previewView.height)
            overlay.updateBoxes(emptyList(), HUD_DET_THRESH, 0)
            overlay.updateCorners(null, null)
            overlay.updateCornerCropRect(null)
        }

        analyzerExecutor = Executors.newSingleThreadExecutor()

        // --- 🔊 Initialize TextToSpeech (local fallback) ---
        tts = TextToSpeech(this) { status: Int ->
            ttsReady = (status == TextToSpeech.SUCCESS)
            Log.i("SudokuTTS", "onInit status=$status ready=$ttsReady")

            if (ttsReady) {
                try {
                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.LOLLIPOP) {
                        val aa = android.media.AudioAttributes.Builder()
                            .setUsage(android.media.AudioAttributes.USAGE_ASSISTANT)
                            .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build()

                        tts.setAudioAttributes(aa)

                        ConversationTelemetry.emit(
                            mapOf(
                                "type" to "TTS_AUDIO_ATTR",
                                "engine" to "AndroidTTS",
                                "usage" to "USAGE_ASSISTANT",
                                "content_type" to "CONTENT_TYPE_SPEECH"
                            )
                        )
                        Log.i("SudokuTTS", "AndroidTTS audio attrs set: USAGE_ASSISTANT + SPEECH")
                    } else {
                        ConversationTelemetry.emit(
                            mapOf(
                                "type" to "TTS_AUDIO_ATTR",
                                "engine" to "AndroidTTS",
                                "usage" to "LEGACY",
                                "content_type" to "LEGACY"
                            )
                        )
                        Log.i("SudokuTTS", "AndroidTTS audio attrs: legacy (<21)")
                    }
                } catch (t: Throwable) {
                    Log.w("SudokuTTS", "Failed to set AndroidTTS audio attributes", t)
                    ConversationTelemetry.emit(
                        mapOf(
                            "type" to "TTS_AUDIO_ATTR_ERROR",
                            "engine" to "AndroidTTS",
                            "message" to (t.message ?: t.toString())
                        )
                    )
                }

                val result = tts.setLanguage(java.util.Locale.getDefault())
                Log.i("SudokuTTS", "setLanguage result=$result")
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.w("SudokuTTS", "Selected TTS language not supported on this device.")
                }

                initTtsListener()

                if (ttsPending.isNotEmpty()) {
                    val copy = ttsPending.toList()
                    ttsPending.clear()
                    copy.forEach { speakAssistant(it) }
                }
            } else {
                Log.e("SudokuTTS", "TTS init failed")
            }
        }

        // --- 🎤 ASR pre-wiring (permission check + prewarm SpeechRecognizer) ---
        val hasMic = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED

        if (hasMic) {
            asr = SudoASR(this).apply {
                prewarm()

                listener = object : SudoASR.Listener {
                    override fun onReady(localeTag: String) {
                        Log.i("SudoASR", "READY locale=$localeTag")
                    }

                    override fun onBegin() {
                        Log.i("SudoASR", "BEGIN")
                    }

                    override fun onRmsChanged(rmsDb: Float) {
                        // Keep invisible in Phase 0/1 (no mic bars)
                    }

                    override fun onPartial(text: String) {
                        // Debug only
                    }

                    override fun onEnd() {
                        Log.i("SudoASR", "END (waiting final)")
                    }

                    override fun onHeard(text: String, confidence: Float?) {
                        // Keep for backwards compatibility / optional debug.
                        val t = text.trim()
                        if (t.isBlank()) return
                        logAsrHeard("HEARD", t, confidence)
                    }

                    override fun onFinal(rowId: Int, text: String, confidence: Float?, reason: String) {
                        val t = text.trim()
                        if (t.isBlank()) {
                            Log.i("SudoASR", "FINAL dropped blank final")
                            return
                        }

                        logAsrHeard("FINAL", t, confidence)
                        turnController.onAsrFinal(t, rowId = rowId, confidence = confidence)

                        if (!USE_CONDUCTOR_ONLY) {
                            handleUserUtterance(t) // legacy
                        } else {
                            // migrated path
                            //sudoStore.dispatch(Evt.AsrFinal(rowId, t, confidence))
                            //sudoStore.dispatch(com.contextionary.sudoku.conductor.Evt.AsrFinal(text = t, confidence = confidence))
                            sudoStore.dispatch(com.contextionary.sudoku.conductor.Evt.AsrFinal(rowId = rowId, text = t, confidence = confidence))
                        }
                    }

                    override fun onError(code: Int) {
                        val name = SudoASR.errorName(code)
                        Log.w("SudoASR", "ERROR code=$code ($name)")
                        turnController.onAsrError(code, name)
                    }
                }
            }

            Log.i("SudoASR", "Prewarmed SudoASR + listener wired")
            wireTurnControllerWorkers()

        } else {
            Log.i("SudoASR", "Mic permission not granted yet; ASR will prompt when needed.")
            Toast.makeText(this, "Enable microphone permission to talk to Sudo.", Toast.LENGTH_SHORT).show()
            wireTurnControllerWorkers()
        }

        runConversationRecoveryBeforeVoiceLoop()
        initAzureTtsIfConfigured()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            askCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }


    private fun ensureSudoStore() {
        if (::sudoStore.isInitialized) return

        val sid: String = runCatching { convoSessionId }
            .getOrNull()
            ?.takeIf { it.isNotBlank() }
            ?: java.util.UUID.randomUUID().toString()

        val policy: com.contextionary.sudoku.conductor.LlmPolicy =
            com.contextionary.sudoku.conductor.policy.CoordinatorPolicyAdapter(
                coord = llmCoordinator,
                systemPrompt = systemPromptText
            )

        var applyEditSeq = 0L

        var lastAssistantSpoken: String = ""

        fun invokeNoArgIfExists(target: Any?, methodName: String): Boolean {
            if (target == null) return false
            return runCatching {
                val m = target::class.java.methods.firstOrNull { it.name == methodName && it.parameterTypes.isEmpty() }
                if (m != null) {
                    m.isAccessible = true
                    m.invoke(target)
                    true
                } else false
            }.getOrDefault(false)
        }

        fun invokeStringArgIfExists(target: Any?, methodName: String, arg: String): Boolean {
            if (target == null) return false
            return runCatching {
                val m = target::class.java.methods.firstOrNull {
                    it.name == methodName &&
                            it.parameterTypes.size == 1 &&
                            it.parameterTypes[0] == String::class.java
                }
                if (m != null) {
                    m.isAccessible = true
                    m.invoke(target, arg)
                    true
                } else false
            }.getOrDefault(false)
        }

        fun requestListenCompat(reason: String) {
            runOnUiThread {
                com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                    mapOf("type" to "REQUEST_LISTEN", "reason" to reason)
                )

                if (isSpeaking || asrSuppressedByTts) {
                    com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                        mapOf(
                            "type" to "REQUEST_LISTEN_SUPPRESSED",
                            "reason" to reason,
                            "isSpeaking" to isSpeaking,
                            "asrSuppressedByTts" to asrSuppressedByTts
                        )
                    )
                    return@runOnUiThread
                }

                val usedTurnController =
                    invokeStringArgIfExists(turnController, "requestListen", reason) ||
                            invokeStringArgIfExists(turnController, "onRequestListen", reason) ||
                            invokeStringArgIfExists(turnController, "startListening", reason)

                if (usedTurnController) {
                    com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                        mapOf("type" to "REQUEST_LISTEN_OK", "via" to "turnController", "reason" to reason)
                    )
                    return@runOnUiThread
                }

                val usedAsr =
                    invokeNoArgIfExists(asr, "start") ||
                            invokeNoArgIfExists(asr, "startListening") ||
                            invokeStringArgIfExists(asr, "start", reason) ||
                            invokeStringArgIfExists(asr, "startListening", reason)

                if (usedAsr) {
                    com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                        mapOf("type" to "REQUEST_LISTEN_OK", "via" to "asr", "reason" to reason)
                    )
                } else {
                    com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                        mapOf("type" to "REQUEST_LISTEN_FAILED", "reason" to reason)
                    )
                    android.util.Log.w(
                        "MainActivity",
                        "Eff.RequestListen: could not find a start/listen method on turnController or asr"
                    )
                }
            }
        }

        /**
         * Single place that:
         * - re-renders SudokuResultView from canonical arrays
         * - recomputes GridState/LLMGridState
         * - emits exactly one GridSnapshotUpdated
         */
        fun emitSnapshotOnce(
            emit: (com.contextionary.sudoku.conductor.Evt) -> Unit,
            reason: String,
            seq: Long
        ) {
            val gs = buildGridStateFromOverlay()
            if (gs != null) {
                val llmGrid = buildLLMGridStateFromOverlay(gs)
                val snap = com.contextionary.sudoku.conductor.GridSnapshot(llm = llmGrid)
                emit(com.contextionary.sudoku.conductor.Evt.GridSnapshotUpdated(snap))

                com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                    mapOf(
                        "type" to "GRID_SNAPSHOT_UPDATED_EMIT",
                        "edit_seq" to seq,
                        "reason" to reason
                    )
                )
            } else {
                com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                    mapOf(
                        "type" to "GRID_SNAPSHOT_UPDATED_MISSED",
                        "edit_seq" to seq,
                        "reason" to "buildGridStateFromOverlay returned null ($reason)"
                    )
                )
                android.util.Log.e("MainActivity", "Gate1 FAIL: buildGridStateFromOverlay() returned null ($reason)")
            }
        }

        fun rerenderFromCanonical() {
            val boldCorrected = BooleanArray(81) { i -> uiAuto[i] || uiManual[i] }

            resultsSudokuView?.setUiData(
                displayDigits = uiDigits,
                displayConfs = uiConfs,
                shownIsGiven = uiGiven,
                shownIsSolution = uiSol,
                candidatesMask = uiCand,
                shownIsAutoCorrected = boldCorrected
            )
        }

        val runner = object : com.contextionary.sudoku.conductor.EffectRunner {

            override fun run(
                effect: com.contextionary.sudoku.conductor.Eff,
                emit: (com.contextionary.sudoku.conductor.Evt) -> Unit
            ) {
                try {
                    when (effect) {

                        is com.contextionary.sudoku.conductor.Eff.UpdateUiMessage -> {
                            runOnUiThread {
                                runCatching { updateSudoMessage(effect.text) }
                                    .onFailure { android.util.Log.w("MainActivity", "UpdateUiMessage failed", it) }
                            }
                        }

                        is com.contextionary.sudoku.conductor.Eff.Speak -> {
                            com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                mapOf(
                                    "type" to "EFFECT_RUN",
                                    "effect" to "Speak",
                                    "text_len" to effect.text.length,
                                    "listen_after" to effect.listenAfter
                                )
                            )

                            lastAssistantSpoken = effect.text

                            runOnUiThread {
                                runCatching { speakAssistant(effect.text, listenAfter = effect.listenAfter) }
                                    .onFailure { android.util.Log.w("MainActivity", "Speak failed", it) }
                            }
                        }

                        // ✅ NEW: Focus highlight (pulsing yellow border)
                        is com.contextionary.sudoku.conductor.Eff.SetFocusCell -> {
                            com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                mapOf(
                                    "type" to "EFFECT_RUN",
                                    "effect" to "SetFocusCell",
                                    "cellIndex" to (effect.cellIndex ?: -1),
                                    "reason" to (effect.reason ?: "")
                                )
                            )
                            runOnUiThread {
                                runCatching {
                                    val v = resultsSudokuView ?: return@runCatching
                                    val idx = effect.cellIndex
                                    if (idx != null && idx in 0..80) {
                                        v.startConfirmationPulse(idx)
                                    } else {
                                        v.stopConfirmationPulse()
                                    }
                                }.onFailure { t ->
                                    android.util.Log.w("MainActivity", "SetFocusCell failed", t)
                                }
                            }
                        }

                        is com.contextionary.sudoku.conductor.Eff.RequestListen -> {
                            com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                mapOf(
                                    "type" to "EFFECT_RUN",
                                    "effect" to "RequestListen",
                                    "reason" to effect.reason
                                )
                            )
                            android.util.Log.i("MainActivity", "Eff.RequestListen(reason=${effect.reason})")
                            requestListenCompat(reason = "eff:${effect.reason}")
                        }

                        is com.contextionary.sudoku.conductor.Eff.StopAsr -> {
                            android.util.Log.i("MainActivity", "Eff.StopAsr(reason=${effect.reason})")
                            com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                mapOf("type" to "EFF_STOP_ASR", "reason" to effect.reason)
                            )

                            runCatching { asr?.stop() }
                                .recoverCatching {
                                    invokeNoArgIfExists(asr, "stop")
                                    invokeNoArgIfExists(asr, "cancel")
                                }
                                .onFailure { android.util.Log.w("MainActivity", "StopAsr failed", it) }
                        }

                        is com.contextionary.sudoku.conductor.Eff.ApplyCellEdit -> {
                            com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                mapOf(
                                    "type" to "EFFECT_RUN",
                                    "effect" to "ApplyCellEdit",
                                    "cellIndex" to effect.cellIndex,
                                    "digit" to effect.digit,
                                    "source" to effect.source
                                )
                            )

                            val seq = ++applyEditSeq
                            val idx = effect.cellIndex
                            val d = effect.digit

                            com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                mapOf(
                                    "type" to "APPLY_CELL_EDIT_BEGIN",
                                    "edit_seq" to seq,
                                    "cellIndex" to idx,
                                    "digit" to d,
                                    "source" to effect.source
                                )
                            )

                            runOnUiThread {
                                runCatching {
                                    if (idx !in 0..80 || d !in 0..9) return@runCatching
                                    if (resultsSudokuView == null || resultsDigits == null || resultsConfidences == null) {
                                        android.util.Log.w("MainActivity", "ApplyCellEdit ignored: results UI not ready")
                                        return@runCatching
                                    }

                                    val old = uiDigits[idx]
                                    uiDigits[idx] = d
                                    uiConfs[idx] = 1.0f

                                    if (d == 0) {
                                        uiGiven[idx] = false
                                        uiSol[idx] = false
                                    }

                                    uiCand[idx] = 0

                                    uiAuto[idx] = false
                                    uiManual[idx] = true

                                    System.arraycopy(uiDigits, 0, resultsDigits!!, 0, 81)
                                    System.arraycopy(uiConfs, 0, resultsConfidences!!, 0, 81)

                                    rerenderFromCanonical()
                                    emitSnapshotOnce(emit, reason = "apply_cell_edit idx=$idx", seq = seq)

                                    if (old != d) {
                                        com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                            mapOf(
                                                "type" to "APPLY_CELL_EDIT_APPLIED",
                                                "edit_seq" to seq,
                                                "cellIndex" to idx,
                                                "from" to old,
                                                "to" to d,
                                                "source" to effect.source
                                            )
                                        )
                                    }
                                }.onFailure { t ->
                                    android.util.Log.w("MainActivity", "ApplyCellEdit failed", t)
                                    com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                        mapOf(
                                            "type" to "APPLY_CELL_EDIT_CRASH",
                                            "edit_seq" to seq,
                                            "err" to (t.message ?: t.toString())
                                        )
                                    )
                                }
                            }
                        }

                        is com.contextionary.sudoku.conductor.Eff.ConfirmCellValue -> {
                            com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                mapOf(
                                    "type" to "EFFECT_RUN",
                                    "effect" to "ConfirmCellValue",
                                    "cellIndex" to effect.cellIndex,
                                    "digit" to effect.digit,
                                    "source" to effect.source,
                                    "changed" to effect.changed
                                )
                            )

                            val seq = ++applyEditSeq
                            val idx = effect.cellIndex
                            val d = effect.digit

                            com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                mapOf(
                                    "type" to "CONFIRM_CELL_VALUE_BEGIN",
                                    "edit_seq" to seq,
                                    "cellIndex" to idx,
                                    "digit" to d,
                                    "source" to effect.source,
                                    "changed" to effect.changed
                                )
                            )

                            runOnUiThread {
                                runCatching {
                                    if (idx !in 0..80 || d !in 0..9) return@runCatching
                                    if (resultsSudokuView == null) return@runCatching

                                    uiConfirmed[idx] = true

                                    emitSnapshotOnce(
                                        emit,
                                        reason = "confirm_cell_value idx=$idx changed=${effect.changed}",
                                        seq = seq
                                    )

                                    com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                        mapOf(
                                            "type" to "CONFIRM_CELL_VALUE_APPLIED",
                                            "edit_seq" to seq,
                                            "cellIndex" to idx,
                                            "digit" to d,
                                            "changed" to effect.changed
                                        )
                                    )
                                }.onFailure { t ->
                                    android.util.Log.w("MainActivity", "ConfirmCellValue failed", t)
                                    com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                        mapOf(
                                            "type" to "CONFIRM_CELL_VALUE_CRASH",
                                            "edit_seq" to seq,
                                            "err" to (t.message ?: t.toString())
                                        )
                                    )
                                }
                            }
                        }

                        is com.contextionary.sudoku.conductor.Eff.ApplyCellClassify -> {
                            val seq = ++applyEditSeq
                            val idx = effect.cellIndex

                            com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                mapOf(
                                    "type" to "APPLY_CELL_CLASSIFY_BEGIN",
                                    "edit_seq" to seq,
                                    "cellIndex" to idx,
                                    "cellClass" to effect.cellClass.name,
                                    "source" to effect.source
                                )
                            )

                            runOnUiThread {
                                runCatching {
                                    if (idx !in 0..80) return@runCatching
                                    if (resultsSudokuView == null) return@runCatching

                                    when (effect.cellClass) {
                                        com.contextionary.sudoku.conductor.CellClass.GIVEN -> {
                                            uiGiven[idx] = true
                                            uiSol[idx] = false
                                        }
                                        com.contextionary.sudoku.conductor.CellClass.SOLUTION -> {
                                            uiGiven[idx] = false
                                            uiSol[idx] = true
                                        }
                                        com.contextionary.sudoku.conductor.CellClass.EMPTY -> {
                                            uiGiven[idx] = false
                                            uiSol[idx] = false
                                            uiDigits[idx] = 0
                                            uiConfs[idx] = 1.0f
                                            uiCand[idx] = 0
                                            uiAuto[idx] = false
                                            uiManual[idx] = true
                                        }
                                    }

                                    rerenderFromCanonical()
                                    emitSnapshotOnce(emit, reason = "apply_cell_classify idx=$idx", seq = seq)
                                }.onFailure { t ->
                                    android.util.Log.w("MainActivity", "ApplyCellClassify failed", t)
                                    com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                        mapOf(
                                            "type" to "APPLY_CELL_CLASSIFY_CRASH",
                                            "edit_seq" to seq,
                                            "err" to (t.message ?: t.toString())
                                        )
                                    )
                                }
                            }
                        }

                        is com.contextionary.sudoku.conductor.Eff.ApplyCellCandidatesMask -> {
                            val seq = ++applyEditSeq
                            val idx = effect.cellIndex
                            val mask = effect.candidateMask

                            runOnUiThread {
                                runCatching {
                                    if (idx !in 0..80) return@runCatching
                                    uiCand[idx] = mask.coerceIn(0, (1 shl 9) - 1)
                                    rerenderFromCanonical()
                                    emitSnapshotOnce(emit, reason = "apply_candidates idx=$idx", seq = seq)
                                }.onFailure { t ->
                                    android.util.Log.w("MainActivity", "ApplyCellCandidatesMask failed", t)
                                }
                            }
                        }

                        is com.contextionary.sudoku.conductor.Eff.CallPolicy -> {
                            com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                mapOf(
                                    "type" to "EFFECT_RUN",
                                    "effect" to "CallPolicy",
                                    "reason" to effect.reason,
                                    "turnId" to effect.turnId,
                                    "mode" to (effect.mode?.name ?: "null"),
                                    "userText_preview" to effect.userText.take(120)
                                )
                            )
                            android.util.Log.w("MainActivity", "Eff.CallPolicy should be handled by SudoStore; ignoring.")
                        }

                        is com.contextionary.sudoku.conductor.Eff.CallPolicyContinuationTick2 -> {
                            com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                mapOf(
                                    "type" to "EFFECT_RUN",
                                    "effect" to "CallPolicyContinuationTick2",
                                    "tool_results_n" to effect.toolResults.size,
                                    "mode" to effect.mode.name,
                                    "reason" to effect.reason,
                                    "turnId" to effect.turnId
                                )
                            )

                            lifecycleScope.launch {
                                val t0 = android.os.SystemClock.elapsedRealtime()

                                // ----------------------------
                                // Minimal helpers (local-safe)
                                // ----------------------------
                                fun anyToString(x: Any?): String? = (x as? String)?.trim()

                                fun argsToJsonObject(args: Map<String, Any?>): org.json.JSONObject {
                                    val o = org.json.JSONObject()
                                    for ((k, v) in args) {
                                        when (v) {
                                            null -> o.put(k, org.json.JSONObject.NULL)
                                            is Boolean, is Int, is Long, is Double, is Float, is String -> o.put(k, v)
                                            is Number -> o.put(k, v.toDouble())
                                            is Map<*, *> -> {
                                                @Suppress("UNCHECKED_CAST")
                                                o.put(k, argsToJsonObject(v as Map<String, Any?>))
                                            }
                                            is List<*> -> {
                                                val arr = org.json.JSONArray()
                                                for (item in v) {
                                                    when (item) {
                                                        null -> arr.put(org.json.JSONObject.NULL)
                                                        is Boolean, is Int, is Long, is Double, is Float, is String -> arr.put(item)
                                                        is Number -> arr.put(item.toDouble())
                                                        is Map<*, *> -> {
                                                            @Suppress("UNCHECKED_CAST")
                                                            arr.put(argsToJsonObject(item as Map<String, Any?>))
                                                        }
                                                        else -> arr.put(item.toString())
                                                    }
                                                }
                                                o.put(k, arr)
                                            }
                                            else -> o.put(k, v.toString())
                                        }
                                    }
                                    return o
                                }

                                // -----------------------------------------
                                // ✅ Robust ToolCall factory via reflection
                                // -----------------------------------------
                                fun toConductorToolCall(wireName: String, args: Map<String, Any?>): com.contextionary.sudoku.conductor.ToolCall? {
                                    val tcClass = com.contextionary.sudoku.conductor.ToolCall::class.java
                                    val argsJson = argsToJsonObject(args)

                                    // Candidate factory method names you likely have somewhere (ToolCall or ToolCall.Companion)
                                    val methodNames = listOf(
                                        "fromWire", "fromNameArgs", "from", "decode", "parse", "fromJson", "fromJSONObject", "fromMap"
                                    )

                                    // 1) Try Companion.INSTANCE methods first (most Kotlin patterns)
                                    try {
                                        val companion = tcClass.declaredClasses.firstOrNull { it.simpleName == "Companion" }
                                        if (companion != null) {
                                            val instField = companion.getDeclaredField("INSTANCE")
                                            instField.isAccessible = true
                                            val inst = instField.get(null)

                                            // Try (String, JSONObject)
                                            for (mn in methodNames) {
                                                val m = companion.methods.firstOrNull { it.name == mn && it.parameterTypes.size == 2 }
                                                if (m != null) {
                                                    val p0 = m.parameterTypes[0]
                                                    val p1 = m.parameterTypes[1]
                                                    val out = when {
                                                        p0 == String::class.java && p1 == org.json.JSONObject::class.java ->
                                                            m.invoke(inst, wireName, argsJson)
                                                        p0 == String::class.java && Map::class.java.isAssignableFrom(p1) ->
                                                            m.invoke(inst, wireName, args)
                                                        else -> null
                                                    }
                                                    if (out is com.contextionary.sudoku.conductor.ToolCall) return out
                                                }
                                            }

                                            // Try (JSONObject) where JSON contains name+args
                                            val packed = org.json.JSONObject()
                                                .put("name", wireName)
                                                .put("args", argsJson)

                                            for (mn in methodNames) {
                                                val m = companion.methods.firstOrNull { it.name == mn && it.parameterTypes.size == 1 && it.parameterTypes[0] == org.json.JSONObject::class.java }
                                                if (m != null) {
                                                    val out = m.invoke(inst, packed)
                                                    if (out is com.contextionary.sudoku.conductor.ToolCall) return out
                                                }
                                            }
                                        }
                                    } catch (_: Throwable) {
                                        // ignore; we’ll try static methods next
                                    }

                                    // 2) Try static methods on ToolCall itself
                                    try {
                                        for (mn in methodNames) {
                                            val m2 = tcClass.methods.firstOrNull { it.name == mn && it.parameterTypes.size == 2 }
                                            if (m2 != null) {
                                                val p0 = m2.parameterTypes[0]
                                                val p1 = m2.parameterTypes[1]
                                                val out = when {
                                                    p0 == String::class.java && p1 == org.json.JSONObject::class.java ->
                                                        m2.invoke(null, wireName, argsJson)
                                                    p0 == String::class.java && Map::class.java.isAssignableFrom(p1) ->
                                                        m2.invoke(null, wireName, args)
                                                    else -> null
                                                }
                                                if (out is com.contextionary.sudoku.conductor.ToolCall) return out
                                            }
                                        }

                                        // Try static (JSONObject) packed
                                        val packed = org.json.JSONObject()
                                            .put("name", wireName)
                                            .put("args", argsJson)

                                        for (mn in methodNames) {
                                            val m1 = tcClass.methods.firstOrNull { it.name == mn && it.parameterTypes.size == 1 && it.parameterTypes[0] == org.json.JSONObject::class.java }
                                            if (m1 != null) {
                                                val out = m1.invoke(null, packed)
                                                if (out is com.contextionary.sudoku.conductor.ToolCall) return out
                                            }
                                        }
                                    } catch (_: Throwable) {
                                        // ignore
                                    }

                                    // Nothing matched
                                    return null
                                }

                                try {
                                    // Build fresh grid snapshot AFTER tools (overlay is canonical now)
                                    val gs = buildGridStateFromOverlay()
                                    if (gs == null) {
                                        com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                            mapOf(
                                                "type" to "POLICY_CONTINUATION_ERR",
                                                "err" to "grid_state_null",
                                                "turnId" to effect.turnId
                                            )
                                        )
                                        emit(com.contextionary.sudoku.conductor.Evt.PolicyContinuationFailed)
                                        return@launch
                                    }
                                    val llmGridAfter = buildLLMGridStateFromOverlay(gs)

                                    val ack = lastAssistantSpoken.trim()

                                    // ✅ Coordinator returns logic.LlmToolCall
                                    val llmTools: List<com.contextionary.sudoku.logic.LlmToolCall> = runCatching {
                                        llmCoordinator.sendToLLMToolsContinuationTick2(
                                            sessionId = sid,
                                            systemPrompt = systemPromptText,
                                            gridStateAfterTools = llmGridAfter,
                                            llm1ReplyText = ack,
                                            toolResults = effect.toolResults,
                                            stateHeader = effect.stateHeader,
                                            continuationUserMessage = "Continue.",
                                            turnId = effect.turnId
                                        )
                                    }.getOrElse { callErr ->
                                        val dt = android.os.SystemClock.elapsedRealtime() - t0
                                        com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                            mapOf(
                                                "type" to "POLICY_CONTINUATION_CRASH",
                                                "ms" to dt,
                                                "turnId" to effect.turnId,
                                                "err" to (callErr.message ?: callErr.toString())
                                            )
                                        )
                                        emit(com.contextionary.sudoku.conductor.Evt.PolicyContinuationFailed)
                                        return@launch
                                    }

                                    val dt = android.os.SystemClock.elapsedRealtime() - t0

                                    // Extract reply text (wire-name is almost certainly "reply")
                                    val replyText = llmTools
                                        .firstOrNull { it.name.equals("reply", ignoreCase = true) }
                                        ?.args
                                        ?.get("text")
                                        ?.toString()
                                        ?.trim()
                                        .orEmpty()

                                    if (replyText.isBlank()) {
                                        com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                            mapOf(
                                                "type" to "POLICY_CONTINUATION_EMPTY",
                                                "ms" to dt,
                                                "turnId" to effect.turnId
                                            )
                                        )
                                        emit(com.contextionary.sudoku.conductor.Evt.PolicyContinuationFailed)
                                        return@launch
                                    }

                                    // Convert to conductor ToolCall using existing factory (via reflection)
                                    val conductorTools = llmTools.mapNotNull { tc ->
                                        toConductorToolCall(tc.name, tc.args)
                                    }

                                    if (conductorTools.isEmpty()) {
                                        com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                            mapOf(
                                                "type" to "POLICY_CONTINUATION_NO_TOOLCALL_FACTORY_MATCH",
                                                "ms" to dt,
                                                "turnId" to effect.turnId,
                                                "llm_tool_names" to llmTools.map { it.name }
                                            )
                                        )
                                        emit(com.contextionary.sudoku.conductor.Evt.PolicyContinuationFailed)
                                        return@launch
                                    }

                                    com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                        mapOf(
                                            "type" to "POLICY_CONTINUATION_OK",
                                            "ms" to dt,
                                            "turnId" to effect.turnId,
                                            "reply_len" to replyText.length,
                                            "tool_n" to conductorTools.size,
                                            // ToolCall may not expose .name; log the LLM wire names instead:
                                            "tool_names" to llmTools.map { it.name }
                                        )
                                    )

                                    // ✅ Most important: emit tools so Store/Conductor continues normally
                                    emit(com.contextionary.sudoku.conductor.Evt.PolicyTools(conductorTools))

                                    // ✅ Optional compatibility event (keep if your store expects it)
                                    emit(com.contextionary.sudoku.conductor.Evt.PolicyContinuationReply(replyText))

                                } catch (t: Throwable) {
                                    val dt = android.os.SystemClock.elapsedRealtime() - t0
                                    android.util.Log.e("MainActivity", "Tick2 crashed (dt=${dt}ms)", t)
                                    com.contextionary.sudoku.telemetry.ConversationTelemetry.emit(
                                        mapOf(
                                            "type" to "POLICY_CONTINUATION_CRASH",
                                            "ms" to dt,
                                            "turnId" to effect.turnId,
                                            "err" to (t.message ?: t.toString())
                                        )
                                    )
                                    emit(com.contextionary.sudoku.conductor.Evt.PolicyContinuationFailed)
                                }
                            }
                        }

                        else -> {
                            android.util.Log.w("MainActivity", "Unhandled effect: ${effect::class.java.simpleName}")
                        }
                    }
                } catch (t: Throwable) {
                    android.util.Log.e("MainActivity", "EffectRunner.run crashed", t)
                }
            }

            override fun applyTools(
                tools: List<com.contextionary.sudoku.conductor.ToolCall>,
                emit: (com.contextionary.sudoku.conductor.Evt) -> Unit
            ) {
                runCatching { emit(com.contextionary.sudoku.conductor.Evt.PolicyTools(tools)) }
                    .onFailure { android.util.Log.e("MainActivity", "applyTools crashed", it) }
            }
        }

        val conductor = com.contextionary.sudoku.conductor.SudoConductor()
        val initial = com.contextionary.sudoku.conductor.SudoState(
            sessionId = sid,
            mode = com.contextionary.sudoku.conductor.SudoMode.FREE_TALK
        )

        sudoStore = com.contextionary.sudoku.conductor.SudoStore(
            initial = initial,
            conductor = conductor,
            policy = policy,
            effects = runner,
            scope = lifecycleScope
        )

        android.util.Log.i("MainActivity", "sudoStore initialized OK (sessionId=$sid)")

        sudoStore.dispatch(com.contextionary.sudoku.conductor.Evt.AppStarted())
        sudoStore.dispatch(com.contextionary.sudoku.conductor.Evt.CameraActive)
    }


    private fun runConversationRecoveryBeforeVoiceLoop() {
        ConversationTelemetry.emit(
            mapOf(
                "type" to "TURN_RECOVERY_BEGIN",
                "convo_session_id" to convoSessionId
            )
        )

        // IMPORTANT:
        // - promptBuilder should be your PromptBuilder instance
        // - turnLifecycle should be YOUR lifecycle manager (NOT ComponentActivity.lifecycle)
        // - turnStore should be your store (InMemoryTurnStore, etc.)
        val pool: List<Any> = listOf(
            convoSessionId,
            persona,
            promptBuilder,
            turnLifecycle,
            turnStore
        )

        val candidateNames = setOf("recover", "run", "resolve", "restore")

        val rc = recovery

        // Search public + declared methods (declared helps if you made it internal/private)
        val methods = (rc.javaClass.methods.asList() + rc.javaClass.declaredMethods.asList())
            .distinctBy { it.name + "#" + it.parameterTypes.joinToString(",") { t -> t.name } }

        val m = methods.firstOrNull { method ->
            if (method.name !in candidateNames) return@firstOrNull false
            val pts = method.parameterTypes
            // Can we satisfy every parameter type from our pool?
            pts.all { pt -> pool.any { arg -> pt.isAssignableFrom(arg.javaClass) } }
        }

        if (m == null) {
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "TURN_RECOVERY_SKIPPED",
                    "reason" to "no_compatible_method_found",
                    "controller" to rc.javaClass.simpleName
                )
            )
            Log.i("TurnRecovery", "No compatible RecoveryController method found; skipping.")
            return
        }

        fun pickArgs(paramTypes: Array<Class<*>>): Array<Any?> {
            val used = BooleanArray(pool.size)
            return paramTypes.map { pt ->
                var picked: Any? = null
                for (i in pool.indices) {
                    if (!used[i] && pt.isAssignableFrom(pool[i].javaClass)) {
                        used[i] = true
                        picked = pool[i]
                        break
                    }
                }
                picked
            }.toTypedArray()
        }

        runCatching {
            m.isAccessible = true
            val args = pickArgs(m.parameterTypes)
            m.invoke(rc, *args)

            ConversationTelemetry.emit(
                mapOf(
                    "type" to "TURN_RECOVERY_OK",
                    "method" to m.name,
                    "controller" to rc.javaClass.simpleName
                )
            )
            Log.i("TurnRecovery", "Recovery invoked via ${rc.javaClass.simpleName}.${m.name}(...)")
        }.onFailure { t ->
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "TURN_RECOVERY_ERROR",
                    "method" to m.name,
                    "controller" to rc.javaClass.simpleName,
                    "message" to (t.message ?: t.toString())
                )
            )
            Log.w("TurnRecovery", "Recovery failed via ${m.name}", t)
        }
    }

    private inline fun <T> List<T>.indexOfFirstIndexed(predicate: (Int, T) -> Boolean): Int {
        for (i in indices) if (predicate(i, this[i])) return i
        return -1
    }






















    override fun onDestroy() {
        // 0) Kill any delayed runnables (e.g., post-speak auto-listen)
        runCatching { mainHandler.removeCallbacksAndMessages(null) }

        // 1) ASR lifecycle (owned by SudoASR, but nullable)
        runCatching { withAsr { it.stop() } }
        runCatching { voiceBars?.stopSpeaking(source = "asr_rms") }
        runCatching { withAsr { it.release() } }
        asr = null

        // 2) Stop any TTS playback and visuals
        runCatching { azureTts?.stop() }
        azureTts = null

        if (::tts.isInitialized) {
            runCatching { tts.stop() }
            runCatching { tts.shutdown() }
        }
        runCatching { voiceBars?.stopSpeaking(source = "tts_cleanup") }

        // 3) Release audio focus
        runCatching { abandonAudioFocus() }

        // 4) Camera / analysis resources
        if (::analyzerExecutor.isInitialized) runCatching { analyzerExecutor.shutdown() }
        if (::shutter.isInitialized) runCatching { shutter.release() }

        // 5) Call super
        super.onDestroy()
    }







    override fun onPause() {
        stopSpeaking()
        super.onPause()
    }



    private fun requestAudioFocus(): Boolean {
        val attrs = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_ASSISTANT)    // speech from an assistant
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .build()

        val req = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
            .setAudioAttributes(attrs)
            .setOnAudioFocusChangeListener { /* no-op for now */ }
            .build()

        focusRequest = req
        val res = audioManager.requestAudioFocus(req)
        return res == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
    }

    private fun abandonAudioFocus() {
        focusRequest?.let { audioManager.abandonAudioFocusRequest(it) }
        focusRequest = null
    }

    @Volatile private var asrSuppressedByTts: Boolean = false

    private fun pauseAsrForTts(reason: String) {
        if (asrSuppressedByTts) return
        asrSuppressedByTts = true

        ConversationTelemetry.emit(mapOf("type" to "ASR_PAUSE_FOR_TTS", "reason" to reason))

        // Contract-safe: suppress + cancel any active ASR row
        runCatching { asr?.setSpeaking(true) }
    }




    private fun initAzureTtsIfConfigured() {
        val key = BuildConfig.AZURE_SPEECH_KEY
        val region = BuildConfig.AZURE_SPEECH_REGION

        if (key.isBlank() || region.isBlank()) {
            Log.i("SudokuTTS", "Azure TTS disabled (missing key/region); using local Android TTS.")
            azureTts = null
            return
        }

        try {
            azureTts = AzureCloudTtsEngine(
                context = this,
                subscriptionKey = key,
                region = region
            ).apply {
                // IMPORTANT: AzureCloudTtsEngine must have:
                //   var onStopped: ((ttsId: Int, source: String) -> Unit)? = null
                onStopped = { ttsId, source ->
                    runOnUiThread {
                        val isPreempt = (source == "pre_new_tts" || source.startsWith("pre_"))

                        // Bars must always stop (truthfulness), even on preempt.
                        voiceBars?.stopSpeaking(
                            source = "tts_azure_stop:$source",
                            truthful = true
                        )

                        if (isPreempt) {
                            // During preemption we are about to speak again.
                            // Do NOT lift suppression and do NOT complete the turn here.
                            ConversationTelemetry.emit(
                                mapOf(
                                    "type" to "AZURE_STOP_CLEANUP_SKIPPED_PREEMPT",
                                    "tts_id" to ttsId,
                                    "source" to source
                                )
                            )
                            return@runOnUiThread
                        }

                        // Terminal cleanup (real end):
                        asrSuppressedByTts = false
                        runCatching { asr?.setSpeaking(false) }

                        // Conductor completion signal
                        turnController.onTtsFinished()

                        ConversationTelemetry.emit(
                            mapOf(
                                "type" to "AZURE_STOP_CLEANUP_RAN",
                                "tts_id" to ttsId,
                                "source" to source
                            )
                        )
                    }
                }
            }

            Log.i("SudokuTTS", "Azure TTS enabled ($region)")
        } catch (t: Throwable) {
            Log.e("SudokuTTS", "Azure TTS init failed; will use local Android TTS", t)
            azureTts = null
        }
    }


    // Known-good SSML for eastus + JennyNeural. Returns (ssml, localeTag).
    private fun buildSsml(text: String): Pair<String, String> {
        val voiceName = "en-US-JennyNeural"
        val localeTag = "en-US"

        val safe = text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")

        val ssml = """
        <speak version="1.0" xml:lang="$localeTag">
          <voice name="$voiceName">
            <prosody rate="medium" pitch="+0st">
              $safe
            </prosody>
          </voice>
        </speak>
    """.trimIndent()

        return Pair(ssml, localeTag)
    }


    // Speak Sudo's message (queued). Milestone 7 will set Locale per utterance.
    // Speak, and optionally auto-listen as soon as speech ends (Azure preferred).
    private fun speakAssistant(message: String, listenAfter: Boolean = true) {

        fun shortPreview(s: String, n: Int = 90): String {
            val oneLine = s.replace("\n", " ").trim()
            return if (oneLine.length <= n) oneLine else oneLine.take(n) + "…"
        }

        fun callerHint(): String {
            val st = Throwable().stackTrace
            val hit = st.firstOrNull { it.className.contains("MainActivity") && it.methodName != "speakAssistant" }
            return if (hit != null) "${hit.methodName}:${hit.lineNumber}" else "unknown"
        }

        // Reflection helper: request listen via TurnController/ASR if method name differs.
        fun invokeNoArgIfExists(target: Any?, methodName: String): Boolean {
            if (target == null) return false
            return runCatching {
                val m = target::class.java.methods.firstOrNull { it.name == methodName && it.parameterTypes.isEmpty() }
                if (m != null) {
                    m.isAccessible = true
                    m.invoke(target)
                    true
                } else false
            }.getOrDefault(false)
        }

        fun invokeStringArgIfExists(target: Any?, methodName: String, arg: String): Boolean {
            if (target == null) return false
            return runCatching {
                val m = target::class.java.methods.firstOrNull {
                    it.name == methodName &&
                            it.parameterTypes.size == 1 &&
                            it.parameterTypes[0] == String::class.java
                }
                if (m != null) {
                    m.isAccessible = true
                    m.invoke(target, arg)
                    true
                } else false
            }.getOrDefault(false)
        }

        fun requestListenCompat(reason: String) {
            // Don’t fight the conductor; this is a compatibility fallback.
            // It should be idempotent if your TurnController ignores duplicate requests.
            runOnUiThread {
                if (isSpeaking || asrSuppressedByTts) {
                    ConversationTelemetry.emit(
                        mapOf(
                            "type" to "REQUEST_LISTEN_SUPPRESSED",
                            "reason" to reason,
                            "isSpeaking" to isSpeaking,
                            "asrSuppressedByTts" to asrSuppressedByTts
                        )
                    )
                    return@runOnUiThread
                }

                val usedTurnController =
                    invokeStringArgIfExists(turnController, "requestListen", reason) ||
                            invokeStringArgIfExists(turnController, "onRequestListen", reason) ||
                            invokeStringArgIfExists(turnController, "startListening", reason)

                if (usedTurnController) {
                    ConversationTelemetry.emit(mapOf("type" to "REQUEST_LISTEN_OK", "via" to "turnController", "reason" to reason))
                    return@runOnUiThread
                }

                val usedAsr =
                    invokeNoArgIfExists(asr, "start") ||
                            invokeNoArgIfExists(asr, "startListening") ||
                            invokeStringArgIfExists(asr, "start", reason) ||
                            invokeStringArgIfExists(asr, "startListening", reason)

                if (usedAsr) {
                    ConversationTelemetry.emit(mapOf("type" to "REQUEST_LISTEN_OK", "via" to "asr", "reason" to reason))
                } else {
                    ConversationTelemetry.emit(mapOf("type" to "REQUEST_LISTEN_FAILED", "reason" to reason))
                    Log.w("MainActivity", "listenAfter fallback: could not find a start/listen method on turnController or asr")
                }
            }
        }

        // -------- Single-flight: never overlap TTS --------
        synchronized(speakLock) {
            if (speakInFlight) {
                // Keep only the latest message (last-write-wins)
                queuedSpeak = Pair(message, listenAfter)
                ConversationTelemetry.emit(
                    mapOf(
                        "type" to "TTS_QUEUED_WHILE_SPEAKING",
                        "len" to message.length,
                        "hash" to message.hashCode(),
                        "preview" to shortPreview(message),
                        "listenAfter" to listenAfter,
                        "caller" to callerHint()
                    )
                )
                return
            }
            speakInFlight = true
        }

        ConversationTelemetry.emit(
            mapOf(
                "type" to "TTS_REQUEST",
                "len" to message.length,
                "hash" to message.hashCode(),
                "preview" to shortPreview(message),
                "listenAfter" to listenAfter,
                "caller" to callerHint(),
                "azure_ready" to (isAzureConfigured() && azureTts?.isReady() == true)
            )
        )

        fun releaseAndMaybeDrainQueue() {
            val next: Pair<String, Boolean>?
            synchronized(speakLock) {
                speakInFlight = false
                next = queuedSpeak
                queuedSpeak = null
            }
            if (next != null) {
                ConversationTelemetry.emit(mapOf("type" to "TTS_DEQUEUED_NEXT"))
                speakAssistant(next.first, next.second)
            }
        }

        if (showCaptions) updateSudoMessage(message)

        // Correlate this assistant speak with the last accepted user ASR row (if any)
        val speakReqId = nextSpeakReqId++
        val replyToRowId = turnController.consumeLastAcceptedRowId()
        if (replyToRowId != null) {
            ConversationTelemetry.emitTurnPair(rowId = replyToRowId, speakReqId = speakReqId)
        }

        // Log exactly what Sudo is about to say (before any engine starts)
        logSudoSay(message, speakReqId = speakReqId, replyToRowId = replyToRowId)

        val (ssml, localeTag) = buildSsml(message)

        val finishOnce = java.util.concurrent.atomic.AtomicBoolean(false)
        val didRealStart = java.util.concurrent.atomic.AtomicBoolean(false)
        val barsStarted = java.util.concurrent.atomic.AtomicBoolean(false)

        fun uiPrepareToSpeak(engineTag: String) {
            isSpeaking = true
            asrSuppressedByTts = true
            runCatching { asr?.setSpeaking(true) }

            pauseAsrForTts(reason = "tts_prepare:$engineTag")
            requestAudioFocus()
            // 🚫 Do NOT start bars here (truthfulness)
        }

        fun uiOnRealPlaybackStart(engineTag: String) {
            if (barsStarted.compareAndSet(false, true)) {
                voiceBars?.startSpeaking(source = "tts_$engineTag", truthful = true)
            }
        }

        fun uiFinishSpeaking(engineTag: String, finishReason: String) {
            if (!finishOnce.compareAndSet(false, true)) return

            isSpeaking = false

            if (barsStarted.compareAndSet(true, false)) {
                voiceBars?.stopSpeaking(source = "tts_$engineTag", truthful = true)
            } else {
                ConversationTelemetry.emit(
                    mapOf(
                        "type" to "UI_BARS_STOP_SKIPPED_NOT_STARTED",
                        "engine" to engineTag,
                        "reason" to finishReason
                    )
                )
            }

            abandonAudioFocus()

            asrSuppressedByTts = false
            runCatching { asr?.setSpeaking(false) }

            // IMPORTANT: onTtsFinished must be fired EXACTLY ONCE per actual playback.
            if (didRealStart.get()) {
                turnController.onTtsFinished()
            } else {
                ConversationTelemetry.emit(
                    mapOf(
                        "type" to "TTS_FINISH_SKIPPED_NOT_STARTED",
                        "engine" to engineTag,
                        "reason" to finishReason
                    )
                )
            }

            // Compatibility fallback:
            // If your conductor relies only on Eff.Speak(listenAfter=true) and forgets to issue Eff.RequestListen,
            // we still kick listening here. If the conductor ALSO issues RequestListen, your TurnController should
            // ignore duplicates (idempotent).
            if (listenAfter) {
                ConversationTelemetry.emit(
                    mapOf(
                        "type" to "LISTEN_AFTER_COMPAT_REQUEST",
                        "engine" to engineTag,
                        "reason" to finishReason
                    )
                )
                requestListenCompat(reason = "listenAfter:$engineTag:$finishReason")
            }

            releaseAndMaybeDrainQueue()
        }

        // Prefer Azure if configured and ready
        if (isAzureConfigured() && azureTts?.isReady() == true) {
            val engine = azureTts!!
            lifecycleScope.launch {
                try {
                    runOnUiThread {
                        uiPrepareToSpeak(engineTag = "azure")
                        Log.i("SudoVoice", "TTS_BEGIN engine=Azure locale=$localeTag")
                    }

                    engine.speakSsml(
                        ssml = ssml,
                        voiceName = "en-US-JennyNeural",
                        localeTag = localeTag,
                        speakReqId = speakReqId,
                        replyToRowId = replyToRowId,

                        onStart = {
                            runOnUiThread {
                                didRealStart.set(true)
                                Log.i("SudoVoice", "TTS_START engine=Azure locale=$localeTag")
                                turnController.onSystemSpeaking("azure_tts_start")
                                uiOnRealPlaybackStart(engineTag = "azure")
                            }
                        },

                        onDone = {
                            runOnUiThread {
                                Log.i("SudoVoice", "TTS_DONE engine=Azure")
                                uiFinishSpeaking(engineTag = "azure", finishReason = "azure_tts_done")
                            }
                        },

                        onError = { err ->
                            runOnUiThread {
                                Log.w("SudoVoice", "TTS_ERROR engine=Azure; falling back to Android", err)
                                uiFinishSpeaking(engineTag = "azure", finishReason = "azure_tts_error")
                                speakWithAndroidTts(
                                    message,
                                    listenAfter = listenAfter,
                                    speakReqId = speakReqId,
                                    replyToRowId = replyToRowId
                                )
                            }
                        }
                    )
                } catch (t: Throwable) {
                    runOnUiThread {
                        Log.w("SudoVoice", "Azure speak threw; falling back to Android", t)
                        uiFinishSpeaking(engineTag = "azure", finishReason = "azure_tts_throw")
                        speakWithAndroidTts(
                            message,
                            listenAfter = listenAfter,
                            speakReqId = speakReqId,
                            replyToRowId = replyToRowId
                        )
                    }
                }
            }
            return
        }

        // Fallback: Android TTS
        speakWithAndroidTts(
            message,
            listenAfter = listenAfter,
            speakReqId = speakReqId,
            replyToRowId = replyToRowId
        )
    }




    private fun iou(a: RectF, b: RectF): Float {
        val ix = maxOf(0f, minOf(a.right, b.right) - maxOf(a.left, b.left))
        val iy = maxOf(0f, minOf(a.bottom, b.bottom) - maxOf(a.top, b.top))
        val inter = ix * iy
        val areaA = a.width() * a.height()
        val areaB = b.width() * b.height()
        val uni = areaA + areaB - inter
        return if (uni <= 0f) 0f else inter / uni
    }

    private fun centerDriftPx(a: RectF, b: RectF): Float {
        val ax = a.centerX(); val ay = a.centerY()
        val bx = b.centerX(); val by = b.centerY()
        val dx = ax - bx; val dy = ay - by
        return kotlin.math.sqrt(dx*dx + dy*dy)
    }

    private fun sizeDriftFrac(a: RectF, b: RectF): Float {
        val dw = kotlin.math.abs(a.width()  - b.width())  / maxOf(1f, b.width())
        val dh = kotlin.math.abs(a.height() - b.height()) / maxOf(1f, b.height())
        return maxOf(dw, dh)
    }



    // MainActivity.kt — ADD THIS if missing
    // Android TTS fallback. Do NOT set your own listener here; initTtsListener() owns it.
    // Android TTS fallback. initTtsListener() owns the TTS listener.
    private fun speakWithAndroidTts(
        text: String,
        listenAfter: Boolean = false,
        speakReqId: Int? = null,
        replyToRowId: Int? = null
    ) {
        if (!ttsReady) {
            Log.w("SudokuTTS", "Android TTS not ready; dropping line")
            return
        }

        val params = Bundle()

        // ✅ Encode correlation ids so initTtsListener can recover them reliably.
        val sr = speakReqId?.toString() ?: "na"
        val rr = replyToRowId?.toString() ?: "na"
        val uttId = "sr${sr}_rr${rr}_${System.currentTimeMillis()}"

        lastUtteranceId = uttId
        pendingListenAfterUtteranceId = if (listenAfter) uttId else null

        // ✅ Suppress ASR (do not force-stop here; listener/conductor handles lifecycle)
        pauseAsrForTts(reason = "android_tts_enqueue")

        ConversationTelemetry.emit(
            mapOf(
                "type" to "TTS_ENQUEUE",
                "engine" to "Android",
                "utterance_id" to uttId,
                "speak_req_id" to speakReqId,
                "reply_to_row_id" to replyToRowId,
                "listen_after" to listenAfter,
                "text_len" to text.length
            )
        )

        Log.i(
            "SudoVoice",
            "TTS_ENQUEUE engine=Android locale=${java.util.Locale.getDefault().toLanguageTag()} uttId=$uttId listenAfter=$listenAfter"
        )

        tts.speak(text, TextToSpeech.QUEUE_FLUSH, params, uttId)
    }



    private fun postMainDelayed(delayMs: Long, block: () -> Unit) {
        mainHandler.postDelayed({ block() }, delayMs)
    }






    private fun stopSpeaking() {
        // Stop Azure first (if present)
        try { azureTts?.stop() } catch (_: Throwable) {}

        // Then stop local Android TTS
        if (::tts.isInitialized) {
            try {
                Log.i("SudokuTTS", "stopSpeaking()")
                tts.stop()
            } catch (_: Throwable) { }
        }

        // Make sure visuals stop even if we didn't get an onDone()
        //voiceBars?.stopSpeaking()
        voiceBars?.stopSpeaking(source = "tts_cleanup")

        // Release audio focus
        abandonAudioFocus()
    }




    // Create the results Overlay (board + buttons) and size/align them precisely.
    // Create the results Overlay (board + buttons) and size/align them precisely.
    private fun ensureResultsOverlay() {
        if (resultsRoot != null) return

        resultsRoot = FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
            alpha = 0f
            isClickable = true
            isFocusable = true
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        }

        // Column centered vertically
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(24.dp(), 24.dp(), 24.dp(), 24.dp())
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        }

        // --- Voice-first header (kept as you had it) ---
        val header = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = 16.dp() }
        }
        voiceBars = SudoVoiceBarsView(this).apply {
            id = View.generateViewId()
            val screenW = resources.displayMetrics.widthPixels
            val targetW = (screenW * 0.40f).toInt()
            layoutParams = LinearLayout.LayoutParams(targetW, 72.dp()).apply {
                bottomMargin = 12.dp()
                gravity = Gravity.CENTER_HORIZONTAL
            }
        }
        header.addView(voiceBars)
        voiceBars?.setMinMax(minFrac = 0.45f, maxFrac = 0.90f)

        sudoMessageTextView = TextView(this).apply {
            id = View.generateViewId()
            text = ""
            setTextColor(Color.parseColor("#DDFFFFFF"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
            gravity = Gravity.CENTER
            maxLines = 2
            ellipsize = TextUtils.TruncateAt.END
            visibility = if (showCaptions) View.VISIBLE else View.GONE
        }
        header.addView(sudoMessageTextView)
        container.addView(header)

        // 1) Board view
        val boardView = SudokuResultView(this).apply {
            id = View.generateViewId()
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0, // height overridden after layout to enforce square
                1f
            ).apply { setMargins(0, 0, 0, 24.dp()) }

            setOnCellClickListener(object : SudokuResultView.OnCellClickListener {
                override fun onCellClicked(row: Int, col: Int) {
                    onOverlayCellClicked(row, col)
                }
            })

            // Long-press anywhere on the board → share PNG
            setOnLongClickListener {
                shareCurrentBoardPng()
                true
            }
        }
        resultsSudokuView = boardView

        // 2) Button row (Retake / Share / Keep)
        val buttonsRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        val retakeBtnCtx = ContextThemeWrapper(this, R.style.Sudoku_Button_Outline)
        val keepBtnCtx   = ContextThemeWrapper(this, R.style.Sudoku_Button_Primary)
        //val shareBtnCtx  = ContextThemeWrapper(this, R.style.Sudoku_Button_Outline)

        val btnRetake = MaterialButton(
            retakeBtnCtx, null, com.google.android.material.R.attr.materialButtonOutlinedStyle
        ).apply {
            text = "Retake"
            isAllCaps = false
            layoutParams = LinearLayout.LayoutParams(0, 52.dp(), 1f).apply {
                setMargins(0, 0, 12.dp(), 0)
            }
            setOnClickListener {
                dismissResults(resumePreview = true)
                resumeAnalyzer()
            }
        }





        val btnKeep = MaterialButton(keepBtnCtx).apply {
            text = "Keep"
            isAllCaps = false
            layoutParams = LinearLayout.LayoutParams(0, 52.dp(), 1f).apply {
                setMargins(12.dp(), 0, 0, 0)
            }
            setOnClickListener { onKeepResults() }
        }

        buttonsRow.addView(btnRetake)
        buttonsRow.addView(btnKeep)

        // 3) Digit picker container (unchanged)
        digitPickerRow = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = 12.dp() }
            visibility = View.INVISIBLE // do not use GONE → avoid layout jumps
        }

        // ... makeDigitChip(...) + rows (unchanged) ...

        // Add everything
        container.addView(boardView)
        container.addView(buttonsRow)
        container.addView(digitPickerRow)
        resultsRoot!!.addView(container)
        (findViewById<ViewGroup>(android.R.id.content)).addView(resultsRoot)

        // After layout: compute centered square & align (unchanged)
        resultsRoot!!.post {
            val rootW = resultsRoot!!.width
            val rootH = resultsRoot!!.height

            val screenMargin   = 24.dp()
            val btnHeight      = 52.dp()
            val pickerHeight   = 40.dp() * 2 + 4.dp()
            val gapAboveBtns   = 24.dp()
            val gapBetweenRows = 12.dp()

            val availW = rootW - 2 * screenMargin
            val availH = rootH - 2 * screenMargin - btnHeight - pickerHeight - gapAboveBtns - gapBetweenRows
            val boardSize = minOf(availW, availH)

            (resultsSudokuView?.layoutParams as LinearLayout.LayoutParams).apply {
                width = boardSize
                height = boardSize
                weight = 0f
                setMargins(0, 0, 0, gapAboveBtns)
                gravity = Gravity.CENTER_HORIZONTAL
            }
            resultsSudokuView?.requestLayout()

            (buttonsRow.layoutParams as LinearLayout.LayoutParams).apply {
                width = boardSize
                height = LinearLayout.LayoutParams.WRAP_CONTENT
                gravity = Gravity.CENTER_HORIZONTAL
            }
            val gridPad = kotlin.math.max((boardSize * 0.04f).toInt(), 16.dp())
            buttonsRow.setPadding(gridPad, 0, gridPad, 0)
            buttonsRow.requestLayout()

            (digitPickerRow?.layoutParams as? LinearLayout.LayoutParams)?.apply {
                width = boardSize
                height = LinearLayout.LayoutParams.WRAP_CONTENT
                gravity = Gravity.CENTER_HORIZONTAL
            }
            digitPickerRow?.setPadding(gridPad, 0, gridPad, 0)
            digitPickerRow?.requestLayout()
        }
    }

    // --- New helper in MainActivity: export current overlay board and share ---
    private fun shareCurrentBoardPng() {
        val view = resultsSudokuView ?: return
        try {
            val bmp = view.renderToBitmap()
            val name = "sudoku_${System.currentTimeMillis()}"
            val uri = PngShareUtils.savePngToCache(
                context = this,
                fileName = "$name.png",
                bmp = bmp
            )
            PngShareUtils.shareImage(this, uri, "Share Sudoku Board")
        } catch (t: Throwable) {
            android.util.Log.e("Share", "Failed to share board", t)
            android.widget.Toast.makeText(this, "Share failed", android.widget.Toast.LENGTH_SHORT).show()
        }
    }


    // Compose & show the result overlay from flat 81-arrays.
// If we already have lastCellReadouts (full 3-head semantics), we delegate to
// showResultsFromReadouts to keep printed-vs-handwritten perfect.



    // Is Azure key/region present at build time?
    private fun isAzureConfigured(): Boolean {
        val key = BuildConfig.AZURE_SPEECH_KEY
        val region = BuildConfig.AZURE_SPEECH_REGION
        return !key.isNullOrBlank() && !region.isNullOrBlank()
    }

    // Minimal SSML escape to avoid breaking XML when using <speak> with raw text.
    private fun ssmlEscape(s: String): String =
        s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")


    private fun onKeepResults() {
        val digits = resultsDigits ?: return

        // Keep the validation mark (you already do it in the Keep click)
        // SessionStore.markValidated()  // <- keep this ONLY in the button handler

        // Option A: do nothing here—overlay already shows the final board.
        // Optionally speak a short confirmation or show a toast:
        Toast.makeText(this, "Saved. You can edit or solve now.", Toast.LENGTH_SHORT).show()

        // If you previously had: startActivity(Intent(this, ResultActivity::class.java))
        // just delete it. Your existing overlay is the “result screen”.
    }








    fun dismissResults(resumePreview: Boolean = true) {
        stopSpeaking()
        sudoMessageTextView?.visibility = View.GONE

        resultsRoot?.visibility = View.GONE
        lastBoardBitmap = null
        lastDigits81 = null

        try { SessionStore.reset() } catch (_: Throwable) {}
        resetUiArrays()

        digitPickerRow?.visibility = View.INVISIBLE
        selectedOverlayIdx = null
        overlayEditable.clear()
        overlayUnresolved.clear()
        resultsSudokuView?.stopConfirmationPulse()

        if (resumePreview) {
            // HARD RESET of capture/detector cycle
            captureLocked = false
            stableCount = 0
            lastDetRect = null
            frameIndex = 0
            lastInferMs = 0L

            // If your overlay uses "locked" visuals, force-clear them here too
            runOnUiThread {
                overlay.setLocked(false)   // if exists in your OverlayView
                overlay.postInvalidateOnAnimation()
            }

            resumeAnalyzer()
        }

        changeGate(GateState.L1, "retake_resume")
        previewView.alpha = 1f
        overlay.alpha = 1f
    }





    private fun initTtsListener() {
        try {
            tts.setOnUtteranceProgressListener(object : android.speech.tts.UtteranceProgressListener() {

                fun parseSpeakReqId(utt: String): Int? {
                    val m = Regex("""\bsr(\d+)\b""").find(utt) ?: return null
                    return m.groupValues[1].toIntOrNull()
                }

                fun parseReplyToRowId(utt: String): Int? {
                    val m = Regex("""\brr(\d+)\b""").find(utt) ?: return null
                    return m.groupValues[1].toIntOrNull()
                }

                override fun onStart(utteranceId: String?) {
                    runOnUiThread {
                        isSpeaking = true

                        asrSuppressedByTts = true
                        runCatching { asr?.setSpeaking(true) }

                        requestAudioFocus()

                        val utt = utteranceId.orEmpty()
                        val listenAfter = (utteranceId != null && utteranceId == pendingListenAfterUtteranceId)

                        val speakReqId = parseSpeakReqId(utt)
                        val replyToRowId = parseReplyToRowId(utt)

                        ConversationTelemetry.emit(
                            mapOf(
                                "type" to "TTS_START",
                                "engine" to "Android",
                                "utterance_id" to utt,
                                "speak_req_id" to speakReqId,
                                "reply_to_row_id" to replyToRowId,
                                "listen_after" to listenAfter
                            )
                        )

                        turnController.onSystemSpeaking("android_tts_start")
                        voiceBars?.startSpeaking(source = "tts_android", truthful = true)

                        Log.i("SudoVoice", "TTS_START engine=Android uttId=$utt listenAfter=$listenAfter sr=$speakReqId rr=$replyToRowId")
                    }
                }

                private fun finishAndroidTts(
                    utteranceId: String?,
                    reason: String,
                    isError: Boolean,
                    errorCode: Int? = null
                ) {
                    runOnUiThread {
                        isSpeaking = false
                        voiceBars?.stopSpeaking(source = "tts_android", truthful = true)

                        abandonAudioFocus()

                        asrSuppressedByTts = false
                        runCatching { asr?.setSpeaking(false) }

                        // ✅ Conductor does cooldown + start listening
                        turnController.onTtsFinished()

                        val utt = utteranceId.orEmpty()
                        val shouldListen = (utteranceId != null && utteranceId == pendingListenAfterUtteranceId)

                        val speakReqId = parseSpeakReqId(utt)
                        val replyToRowId = parseReplyToRowId(utt)

                        val base = mutableMapOf<String, Any?>(
                            "type" to (if (isError) "TTS_ERROR" else "TTS_DONE"),
                            "engine" to "Android",
                            "utterance_id" to utt,
                            "speak_req_id" to speakReqId,
                            "reply_to_row_id" to replyToRowId,
                            "listen_after" to shouldListen,
                            "reason" to reason
                        )
                        if (errorCode != null) base["error_code"] = errorCode
                        ConversationTelemetry.emit(base)

                        if (isError) {
                            Log.w("SudoVoice", "TTS_ERROR engine=Android code=${errorCode ?: -1} uttId=$utt listenAfter=$shouldListen")
                        } else {
                            Log.i("SudoVoice", "TTS_DONE engine=Android uttId=$utt listenAfter=$shouldListen")
                        }

                        if (shouldListen) {
                            pendingListenAfterUtteranceId = null
                            ConversationTelemetry.emit(
                                mapOf(
                                    "type" to "LISTEN_AFTER_DEFERRED_TO_CONDUCTOR",
                                    "engine" to "Android",
                                    "utterance_id" to utt,
                                    "speak_req_id" to speakReqId,
                                    "reply_to_row_id" to replyToRowId,
                                    "reason" to "android_tts_finished"
                                )
                            )
                        }
                    }
                }

                override fun onDone(utteranceId: String?) {
                    finishAndroidTts(utteranceId, reason = "android_tts_done", isError = false)
                }

                @Suppress("OVERRIDE_DEPRECATION")
                override fun onError(utteranceId: String?) {
                    finishAndroidTts(utteranceId, reason = "android_tts_error_deprecated", isError = true)
                }

                override fun onError(utteranceId: String?, errorCode: Int) {
                    finishAndroidTts(utteranceId, reason = "android_tts_error", isError = true, errorCode = errorCode)
                }
            })
        } catch (t: Throwable) {
            Log.e("SudokuTTS", "Failed to set TTS listener", t)
            ConversationTelemetry.emit(
                mapOf(
                    "type" to "TTS_LISTENER_ERROR",
                    "engine" to "Android",
                    "message" to (t.message ?: t.toString())
                )
            )
        }
    }


    private fun resumeAnalyzer() {

        turnController.onCameraActive()

        // Fully reset all scan gating
        locked = false
        handoffInProgress = false

        overlay.setLocked(false)

        // Put the HUD back into a neutral/ready state
        // (Going to L1 tends to be clearer than NONE after a green L3)
        changeGate(GateState.L1, "retake_resume")

        // If you keep any last-corners / last-ROI cached for drawing,
        // clear them here (only if you actually have these methods/fields):
        // overlay.setCorners(null)
        // overlay.setWarp(null)

        previewView.alpha = 1f
        overlay.alpha = 1f

        overlay.postInvalidateOnAnimation()
    }




    /**
     * Build a GridState snapshot from the current overlay / result grid,
     * for the conversation layer (GridConversationCoordinator).
     *
     * - Uses resultsDigits / resultsConfidences as the canonical board.
     * - Uses overlayUnresolved as the set of "still doubtful" cells.
     * - Uses lastAutoCorrectionResult.changedIndices for wasChangedByLogic.
     * - Marks low-confidence cells using LOWCONF_CELL_THR.
     */
    private fun buildGridStateFromOverlay(): GridState? {
        val digitsNow = resultsDigits ?: return null
        val confsNow  = resultsConfidences ?: return null
        val auto      = lastAutoCorrectionResult ?: return null

        val digitsCopy = digitsNow.copyOf()
        val confsCopy  = confsNow.copyOf()

        val unresolvedIndices = overlayUnresolved.toList()

        val conflictIndices = findConflictIndicesForDigits(digitsCopy)
        val isStructurallyValid = conflictIndices.isEmpty()

        val solutionCountCapped = if (isStructurallyValid) {
            sudokuSolver.countSolutions(digitsCopy, maxCount = 2)
        } else 0

        val hasUniqueSolution = (solutionCountCapped == 1)
        val hasMultipleSolutions = (solutionCountCapped >= 2)
        val hasNoSolution = isStructurallyValid && (solutionCountCapped == 0)

        val conflictSet   = conflictIndices.toHashSet()
        val unresolvedSet = unresolvedIndices.toHashSet()

        val autoChangedSet = auto.changedIndices.toHashSet()
        val manualSet = (0 until 81).filter { uiManual[it] }.toHashSet()

        // Pre-autocorrect baseline (what was originally scanned/chosen by decidePick)
        // This assumes you kept lastGridPrediction from attemptRectifyAndClassify()
        val scannedDigits: IntArray? = lastGridPrediction?.digits
        val scannedConfs: FloatArray? = lastGridPrediction?.confidences

        // Auto-corrected target digits right after autocorrect (before manual edits)
        val autoDigits: IntArray = auto.correctedGrid.digits

        val cells = (0 until 81).map { idx ->
            val row = idx / 9
            val col = idx % 9

            val digit = digitsCopy[idx]
            val conf  = confsCopy[idx]

            val scannedD = scannedDigits?.getOrNull(idx)
            val scannedC = scannedConfs?.getOrNull(idx)

            val wasAuto = autoChangedSet.contains(idx)
            val wasManual = uiManual[idx]

            GridCellState(
                index = idx,
                row = row,
                col = col,
                digit = digit,
                confidence = conf,

                isConflict = conflictSet.contains(idx),
                wasChangedByLogic = wasAuto,
                wasManuallyCorrected = wasManual,
                isUnresolved = unresolvedSet.contains(idx),
                isLowConfidence = (digit != 0 && conf < SudokuConfidence.THRESH_HIGH),

                scannedDigit = scannedD,
                scannedConfidence = scannedC,

                autoFromDigit = scannedD,
                autoToDigit = autoDigits.getOrNull(idx),

                manualFromDigit = if (wasManual) uiManualFrom[idx] else null,
                manualToDigit   = if (wasManual) uiManualTo[idx] else null
            )
        }

        return GridState(
            cells = cells,
            digits = digitsCopy,
            confidences = confsCopy,
            conflictIndices = conflictIndices,
            changedByLogic = auto.changedIndices,
            unresolvedIndices = unresolvedIndices,

            manuallyCorrected = manualSet.toList(),

            isStructurallyValid = isStructurallyValid,
            hasUniqueSolution = hasUniqueSolution,
            hasMultipleSolutions = hasMultipleSolutions,
            hasNoSolution = hasNoSolution,
            solutionCountCapped = solutionCountCapped
        )
    }





    /**
     * For the LLM
     * Local copy of the conflict finder used in SudokuAutoCorrector.
     * Returns indices (0..80) of cells that participate in any row/col/box conflict.
     */
    private fun findConflictIndicesForDigits(digits: IntArray): List<Int> {
        val conflicts = mutableSetOf<Int>()

        // Rows
        for (row in 0 until 9) {
            val seen = mutableMapOf<Int, MutableList<Int>>() // digit -> indices
            for (col in 0 until 9) {
                val idx = row * 9 + col
                val v = digits[idx]
                if (v == 0) continue
                val list = seen.getOrPut(v) { mutableListOf() }
                list.add(idx)
            }
            for ((_, idxList) in seen) {
                if (idxList.size > 1) conflicts.addAll(idxList)
            }
        }

        // Columns
        for (col in 0 until 9) {
            val seen = mutableMapOf<Int, MutableList<Int>>()
            for (row in 0 until 9) {
                val idx = row * 9 + col
                val v = digits[idx]
                if (v == 0) continue
                val list = seen.getOrPut(v) { mutableListOf() }
                list.add(idx)
            }
            for ((_, idxList) in seen) {
                if (idxList.size > 1) conflicts.addAll(idxList)
            }
        }

        // 3x3 boxes
        for (boxRow in 0 until 3) {
            for (boxCol in 0 until 3) {
                val seen = mutableMapOf<Int, MutableList<Int>>()
                val startRow = boxRow * 3
                val startCol = boxCol * 3
                for (dr in 0 until 3) {
                    for (dc in 0 until 3) {
                        val row = startRow + dr
                        val col = startCol + dc
                        val idx = row * 9 + col
                        val v = digits[idx]
                        if (v == 0) continue
                        val list = seen.getOrPut(v) { mutableListOf() }
                        list.add(idx)
                    }
                }
                for ((_, idxList) in seen) {
                    if (idxList.size > 1) conflicts.addAll(idxList)
                }
            }
        }

        return conflicts.toList().sorted()
    }




    // Called when the user taps a cell on the overlay SudokuResultView.
    private fun onOverlayCellClicked(row: Int, col: Int) {
        val idx = row * 9 + col

        // Allow taps on any cell that the logic layer marked as editable:
        // editable = unresolved ∪ changed (set up in showResults()).
        if (!overlayEditable.contains(idx)) {
            return
        }

        // Keep a local index of the currently-selected overlay cell
        selectedOverlayIdx = idx

        // Show the bottom digit picker strip
        digitPickerRow?.visibility = View.VISIBLE
    }


    private fun wireTurnControllerWorkers() {
        turnController.attachWorkers(
            startAsr = {
                Log.i("SudoASR", "CMD_START_ASR invoked")

                val a = asr
                if (a == null) {
                    Log.w("SudoASR", "CMD_START_ASR ignored: asr is null (mic permission?)")
                    ConversationTelemetry.emit(
                        mapOf(
                            "type" to "ASR_CMD_START_SKIPPED",
                            "reason" to "asr_null"
                        )
                    )
                    return@attachWorkers
                }

                // Optional defensive check: don't start if we *know* we're speaking/suppressed.
                // (If SudoASR internally enforces this, keep or remove—either is fine.)
                if (isSpeaking) {
                    Log.w("SudoASR", "CMD_START_ASR blocked: isSpeaking=true")
                    ConversationTelemetry.emit(
                        mapOf(
                            "type" to "ASR_CMD_START_SKIPPED",
                            "reason" to "isSpeaking_true"
                        )
                    )
                    return@attachWorkers
                }

                runCatching {
                    // Use your real API — keep asr?.start() if that's correct.
                    a.start()
                }.onFailure {
                    Log.e("SudoASR", "CMD_START_ASR failed", it)
                    ConversationTelemetry.emit(
                        mapOf(
                            "type" to "ASR_CMD_START_ERROR",
                            "msg" to (it.message ?: it.toString())
                        )
                    )
                }
            },
            stopAsr = {
                Log.i("SudoASR", "CMD_STOP_ASR invoked")

                val a = asr
                if (a == null) {
                    ConversationTelemetry.emit(
                        mapOf(
                            "type" to "ASR_CMD_STOP_SKIPPED",
                            "reason" to "asr_null"
                        )
                    )
                    return@attachWorkers
                }

                runCatching {
                    a.stop()
                }.onFailure {
                    ConversationTelemetry.emit(
                        mapOf(
                            "type" to "ASR_CMD_STOP_ERROR",
                            "msg" to (it.message ?: it.toString())
                        )
                    )
                }
            }
        )


        //Log.i("TurnController", "Workers attached (asr_is_null=${asr == null})")
    }





    // -------------------------------------------------------------------------
    // Drop-in replacement: startCamera()
    //  - Amber after "dots appear" dwell (delegated to GateController via gridizedOk)
    //  - Best-of-N: score the last 4 passing frames; lock the best (uses current bmp for work)
    //  - Provisional Green at lock; heavy work happens in attemptRectifyAndClassify()
    //  - Demotions: NONE when no pick; L1 when intersections/gates break
    // -------------------------------------------------------------------------
    private fun startCamera() {

        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()

            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(UiSize(960, 720))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setOutputImageRotationEnabled(true)
                .setTargetRotation(previewView.display.rotation)
                .build()

            // Analyzer core
            analysis.setAnalyzer(analyzerExecutor) { proxy ->
                try {
                    if (locked || handoffInProgress) { proxy.close(); return@setAnalyzer }

                    frameIndex++
                    if (frameIndex % (skipEvery + 1) != 0) { proxy.close(); return@setAnalyzer }

                    val now = SystemClock.elapsedRealtime()
                    if (now - lastInferMs < minInferIntervalMs) { proxy.close(); return@setAnalyzer }
                    lastInferMs = now

                    val bmp = proxy.toBitmapRGBA() ?: run { proxy.close(); return@setAnalyzer }

                    // 1) Detector
                    val dets = detector.infer(bmp, scoreThresh = HUD_DET_THRESH, maxDets = HUD_MAX_DETS)
                    val cxImg = bmp.width / 2f
                    val cyImg = bmp.height / 2f
                    val picked = dets.minByOrNull { det ->
                        val dx = det.box.centerX() - cxImg
                        val dy = det.box.centerY() - cyImg
                        dx * dx + dy * dy
                    }

                    // HUD: boxes
                    runOnUiThread {
                        overlay.setSourceSize(bmp.width, bmp.height)
                        overlay.updateBoxes(if (picked != null) listOf(picked) else emptyList(), HUD_DET_THRESH, HUD_MAX_DETS)
                        overlay.updateCornerCropRect(null)
                        overlay.updateIntersections(null)
                    }

                    if (picked == null) {
                        lastDetRect = null
                        stableCount = 0
                        changeGate(GateState.NONE, "no_detection")
                        proxy.close(); return@setAnalyzer
                    }

                    val roi = RectF(picked.box)

                    // 2) Cyan-guard clearance (in source space)
                    val guardSrc = overlay.getGuardRectInSource()
                    val tolSrc = (min(bmp.width, bmp.height) / 120f).coerceAtLeast(2f)
                    val guardOk = if (guardSrc != null) !touchesBorder(roi, guardSrc, tolSrc) else true
                    if (!guardOk) {
                        stableCount = 0
                        lastDetRect = roi
                        changeGate(GateState.L1, "guard_touch")
                        proxy.close(); return@setAnalyzer
                    }

                    // 3) Temporal stability vs previous detection
                    val stableNow = lastDetRect?.let { prev ->
                        val i = iou(prev, roi)
                        val driftPx = centerDriftPx(prev, roi)
                        val sDrift = sizeDriftFrac(prev, roi)
                        (i >= IOU_THR && driftPx <= CENTER_DRIFT_PX && sDrift <= SIZE_DRIFT_FRAC)
                    } ?: false

                    if (stableNow) stableCount++ else stableCount = 1
                    lastDetRect = roi

                    if (stableCount >= 2 && gateState == GateState.L1) changeGate(GateState.L2, "stable_seen")
                    if (gateState == GateState.NONE) changeGate(GateState.L1, "got_detection")

                    // 4) Lock when we've reached the target stability window
                    if (stableCount >= STABLE_FRAMES) {
                        changeGate(GateState.L3, "lock_on_stability")
                        runOnUiThread { overlay.setLocked(true) }

                        attemptRectifyAndClassify(
                            srcBmp = bmp,
                            detectorRoi = picked.box.toAndroidRect()
                        )
                    }
                } catch (t: Throwable) {
                    Log.e("Detector", "Analyzer error on frame $frameIndex", t)
                    runOnUiThread {
                        overlay.setSourceSize(previewView.width, previewView.height)
                        overlay.updateBoxes(emptyList(), HUD_DET_THRESH, 0)
                        overlay.updateCornerCropRect(null)
                        overlay.updateIntersections(null)
                        overlay.setLocked(false)
                    }
                    lastDetRect = null
                    stableCount = 0
                    changeGate(GateState.NONE, "exception")
                } finally {
                    proxy.close()
                }
            }

            provider.unbindAll()
            provider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )

            // ✅ Notify camera active only once the pipeline is actually bound/active.
            turnController.onCameraActive()

        }, ContextCompat.getMainExecutor(this))
    }






    // --- Safe ROI clip + crop helpers (prevents Bitmap.createBitmap OOB crashes) ---
    private fun clipRectToBitmap(r: Rect, bmp: Bitmap): Rect? {
        if (bmp.width <= 1 || bmp.height <= 1) return null

        val x0 = r.left.coerceIn(0, bmp.width - 1)
        val y0 = r.top.coerceIn(0, bmp.height - 1)
        val x1 = r.right.coerceIn(0, bmp.width)   // right/bottom as exclusive bounds
        val y1 = r.bottom.coerceIn(0, bmp.height)

        // Require a real area (at least 2x2) to keep later math healthy
        if (x1 - x0 < 2 || y1 - y0 < 2) return null

        return Rect(x0, y0, x1, y1)
    }

    private fun safeCropBitmap(src: Bitmap, r: Rect): Bitmap? {
        val rr = clipRectToBitmap(r, src) ?: return null
        return try {
            Bitmap.createBitmap(src, rr.left, rr.top, rr.width(), rr.height())
        } catch (_: Throwable) {
            null
        }
    }







    // Center-crop a square bitmap to an inner region, then resize back to outSize×outSize.
// innerFrac is the fraction to trim from EACH side (e.g. 0.10f = 10% per side).
    private fun centerCropAndResize(
        src: Bitmap,
        innerFrac: Float,
        outSize: Int
    ): Bitmap {
        val w = src.width
        val h = src.height

        // We expect square tiles, but be defensive
        val marginX = ((w * innerFrac).toInt()).coerceIn(0, w / 4)
        val marginY = ((h * innerFrac).toInt()).coerceIn(0, h / 4)

        val cropX = marginX
        val cropY = marginY
        val cropW = (w - 2 * marginX).coerceAtLeast(1)
        val cropH = (h - 2 * marginY).coerceAtLeast(1)

        val cropped = Bitmap.createBitmap(src, cropX, cropY, cropW, cropH)
        val scaled  = Bitmap.createScaledBitmap(cropped, outSize, outSize, true)

        // We don’t need the intermediate cropped bitmap after scaling
        if (cropped != src) {
            try { cropped.recycle() } catch (_: Throwable) {}
        }

        return scaled
    }




    // Runs on a worker thread: expand+crop ROI; call Rectifier.process() to get tiles; push tiles into DigitClassifier; flatten to 81 digits and probabilities. If successful, set 'locked', flash/shutter, and navigate to results; otherwise keep scanning.
    // Runs on a worker thread: use the 10×10 intersection lattice to cut each cell
    // from its own quadrilateral, robust to bowed or bent grid lines.
    // Use the exact expanded ROI (expandedRoiSrc) that the intersections model used.
// This keeps point->ROI-local mapping aligned and fixes the cyan points offset.



    private fun attemptRectifyAndClassify(
        srcBmp: Bitmap,
        detectorRoi: android.graphics.Rect
    ) {
        if (handoffInProgress) return
        handoffInProgress = true

        Thread {
            var err: String? = null
            var lockedLocal = false

            var boardBmpOut: Bitmap? = null

            // Raw readouts from CellInterpreter (pre-autocorrect)
            var readouts9x9: Array<Array<CellReadout>>? = null

            // Corrected UI state to display
            var correctedReadouts9x9: Array<Array<CellReadout>>? = null
            var digitsFlat: IntArray? = null
            var probsFlat: FloatArray? = null
            var givenFlags: BooleanArray? = null
            var solFlags: BooleanArray? = null
            var autoFlags: BooleanArray? = null
            var candMaskFlat: IntArray? = null

            try {
                // --- Parity/debug options ---
                val opts = Rectifier.Options(
                    tileSize = 64,
                    shrink = 0.0f,
                    precompress = true,
                    targetKb = 120,
                    maxSide = 1600,
                    minJpegQuality = 45,
                    robust = true
                )

                // ✅ Pass caseId 3rd, options 4th (named) to match the single signature
                val caseDir = saveRoiForParity(
                    fullFrame = srcBmp,
                    detectorRoi = detectorRoi,
                    caseId = System.currentTimeMillis().toString(),
                    options = opts
                )
                Log.i("Rectify", "caseDir=${caseDir.absolutePath}")

                // 1) Safe crop of the detector ROI (this becomes our "board")
                val boardRoi = safeCropBitmap(srcBmp, detectorRoi)
                if (boardRoi == null) {
                    err = "ROI outside image bounds"
                    runOnUiThread {
                        overlay.setLocked(false)
                        changeGate(GateState.L2, "roi_invalid")
                        Toast.makeText(this, "Rectify failed: ROI invalid", Toast.LENGTH_SHORT).show()
                    }
                    return@Thread
                }

                // 2) TEMP tiling: 9×9 equal tiles + gentle inner crop to mimic shrink
                val tiles = Array(9) { r ->
                    Array(9) { c ->
                        val x0 = (c * boardRoi.width) / 9
                        val y0 = (r * boardRoi.height) / 9
                        val x1 = ((c + 1) * boardRoi.width) / 9
                        val y1 = ((r + 1) * boardRoi.height) / 9
                        val w  = (x1 - x0).coerceAtLeast(1)
                        val h  = (y1 - y0).coerceAtLeast(1)
                        val raw = Bitmap.createBitmap(boardRoi, x0, y0, w, h)
                        centerCropAndResize(raw, innerFrac = 0.0f, outSize = 64)
                    }
                }

                Log.i("Rectify", "tiles=9x9 size=${tiles[0][0].width}x${tiles[0][0].height} innerCrop")

                runCatching {
                    val cellsDir = java.io.File(caseDir, "cells").apply { mkdirs() }
                    for (r in 0 until 9) for (c in 0 until 9) {
                        val f = java.io.File(cellsDir, "cell_${r}_${c}.png")
                        java.io.FileOutputStream(f).use { out ->
                            tiles[r][c].compress(Bitmap.CompressFormat.PNG, 100, out)
                        }
                    }
                    Log.i("Rectify", "saved 81 cells to ${cellsDir.absolutePath}")
                }.onFailure { t ->
                    Log.w("Rectify", "saving cells failed", t)
                }

                // 3) Run CellInterpreter on 9×9 tiles  →  9×9 CellReadout
                ensureCellInterpreter()
                val grid: Array<Array<CellReadout>> = cellInterpreter!!.interpretTiles(tiles)
                readouts9x9 = grid

                // 4) Build autocorrect inputs USING decidePick() (single policy!)
                val inputs = buildAutocorrectInputsFromReadouts(grid)

                // ✅ always log the detailed verifier output
                val verify = verifyAutocorrectInputMatchesCaptured(grid, inputs, AUTOCORR_LOWCONF_THR)
                Log.i("SudokuLogic", verify)

                // Strict verifier: ensure inputs == decidePick(grid) + candidates
                if (!verifyAutocorrectInputsAgainstReadouts(grid, inputs)) {
                    err = "AutocorrectInputs mismatch vs captured readouts (decidePick policy). Aborting autocorrect."
                    Log.e("SudokuLogic", err!!)
                    throw IllegalStateException(err)
                }

                val prediction = GridPrediction.fromFlat(
                    digits = inputs.digits,
                    confidences = inputs.confidences,
                    lowConfThreshold = AUTOCORR_LOWCONF_THR
                )
                lastGridPrediction = prediction

                // --- DEBUG: dump interpreter heads vs chosen input-to-autocorrect ---
                dumpInterpreterDebug(
                    caseDir = caseDir,
                    grid = grid,
                    chosenDigits = prediction.digits,
                    chosenConfs = prediction.confidences
                )

                // 5) Run autocorrect
                // IMPORTANT: classProbs here are the PICKED HEAD probs when available (else heuristic fallback)
                val auto = sudokuAutoCorrector.autoCorrect(
                    prediction = prediction,
                    classProbs = inputs.classProbs
                )
                lastAutoCorrectionResult = auto

                val correctedDigits = auto.correctedGrid.digits

                // Display confidence should reflect the chosen head distribution (and thus autocorrect decisions).
                // computeDisplayConfsFromClassProbs reads p[d] from inputs.classProbs.
                val correctedConfs = computeDisplayConfsFromClassProbs(correctedDigits, inputs.classProbs)


                // Build AUTOCORRECTED provenance flags from changedIndices
                val asAuto = buildAutoFlags(auto)


                // Then GIVEN/SOLUTION flags but excluding auto-changed cells
                val (asGiven, asSol) = buildGivenSolFlagsFromHeadsExcludingAuto(
                    correctedDigits = correctedDigits,
                    headsFlat = inputs.headsFlat,
                    autoFlags = asAuto
                )

                givenFlags = asGiven
                solFlags   = asSol
                autoFlags  = asAuto

                correctedReadouts9x9 = applyAutocorrectToReadouts(
                    original = grid,
                    correctedDigits = correctedDigits,
                    correctedConfs = correctedConfs,
                    headsFlat = inputs.headsFlat
                )

                // Keep for UI display
                boardBmpOut  = boardRoi
                digitsFlat   = correctedDigits
                probsFlat    = correctedConfs
                givenFlags   = asGiven
                solFlags     = asSol
                autoFlags    = asAuto
                candMaskFlat = inputs.candMask

                // 6) Debug artifacts for autocorrect
                dumpAutoCorrectDebug(
                    caseDir = caseDir,
                    prediction = prediction,
                    auto = auto,
                    inputs = inputs,
                    correctedConfs = correctedConfs
                )

                // 7) Save to SessionStore (still keeps full 3-head detail)
                try {
                    val flatReadouts = Array(81) { i ->
                        val r = i / 9
                        val c = i % 9
                        correctedReadouts9x9!![r][c]
                    }
                    SessionStore.ingestCapture(
                        runPath = caseDir.absolutePath,
                        readouts = flatReadouts,
                        p10x10 = null,
                        tiles = null
                    )
                } catch (t: Throwable) {
                    android.util.Log.w("SessionStore", "ingestCapture failed (non-fatal)", t)
                }

                // 8) Optional snapshot log
                try {
                    if (SessionStore.isReadyForValidation()) {
                        val snap = SessionStore.snapshot()
                        android.util.Log.i(
                            "SessionStore",
                            "snapshot: digitsForDisplay.size=${snap.digitsForDisplay.size} nonZero=${snap.nonZeroDigitsCount}"
                        )
                    }
                } catch (t: Throwable) {
                    android.util.Log.w("SessionStore", "snapshot() failed (non-fatal)", t)
                }

                // 9) Post checks for provisional green (use corrected confidences)
                val avgConf = probsFlat!!.average().toFloat()
                val lowConfCells = probsFlat!!.count { it < MIN_AVG_CELL_CONF }
                val postSnap = GateSnapshot(
                    hasDetectorLock = true,
                    gridizedOk      = true,
                    validPoints     = 100,
                    jitterPx128     = 0f,
                    rectifyOk       = true,
                    avgConf         = avgConf,
                    lowConfCells    = lowConfCells
                )
                val sm = gate.update(postSnap)
                runOnUiThread { changeGate(sm, "postRectify") }

                locked = true
                lockedLocal = true

            } catch (t: Throwable) {
                err = t.message ?: "$t"
                Log.e("MainActivity", "attemptRectifyAndClassify failed", t)
            } finally {
                handoffInProgress = false
                if (lockedLocal &&
                    boardBmpOut != null &&
                    correctedReadouts9x9 != null &&
                    digitsFlat != null &&
                    probsFlat != null &&
                    givenFlags != null &&
                    solFlags != null &&
                    autoFlags != null &&
                    candMaskFlat != null
                ) {
                    runOnUiThread {
                        try {
                            // Keep lastCellReadouts aligned with what the user sees (post-autocorrect)
                            lastCellReadouts = correctedReadouts9x9
                            resultsDigits = digitsFlat
                            resultsConfidences = probsFlat

                            showResultsFromReadouts(
                                boardBitmap = boardBmpOut!!,
                                readouts9x9 = correctedReadouts9x9!!,
                                uiDigitsIn = digitsFlat!!,
                                uiConfsIn  = probsFlat!!,
                                uiGivenIn  = givenFlags!!,
                                uiSolIn    = solFlags!!,
                                uiAutoIn   = autoFlags!!,
                                uiCandIn   = candMaskFlat!!
                            )
                        } catch (t: Throwable) {
                            Log.e("MainActivity", "CRASH in showResultsFromReadouts()", t)

                            // Demote gate + unlock so you can keep using the app
                            overlay.setLocked(false)
                            changeGate(GateState.L2, "showResults_crash")

                            Toast.makeText(
                                this,
                                "UI crash after capture: ${t.javaClass.simpleName}: ${t.message}",
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    }
                } else {
                    runOnUiThread {
                        overlay.setLocked(false)
                        changeGate(GateState.L2, "demote_after_fail")
                        if (err != null) {
                            Toast.makeText(this, "Rectify failed: $err", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            }
        }.start()
    }






    private fun showResultsFromReadouts(
        boardBitmap: Bitmap,
        readouts9x9: Array<Array<CellReadout>>,
        uiDigitsIn: IntArray,
        uiConfsIn: FloatArray,
        uiGivenIn: BooleanArray,
        uiSolIn: BooleanArray,
        uiAutoIn: BooleanArray,
        uiCandIn: IntArray
    ) {
        ensureResultsOverlay()

        runCatching {
            resultsSudokuView?.setFromCellGrid(CellGridReadout(cells = readouts9x9))
        }


        // ✅ "Corrected" highlight = auto OR manual (manual likely false at capture time,
// but this keeps the meaning consistent across the app)
        val correctedShown = BooleanArray(81) { i -> uiAutoIn[i] || uiManual[i] }

        resultsSudokuView?.setUiData(
            displayDigits        = uiDigitsIn,
            displayConfs         = uiConfsIn,
            shownIsGiven         = uiGivenIn,
            shownIsSolution      = uiSolIn,
            candidatesMask       = uiCandIn,
            shownIsAutoCorrected = correctedShown
        )

        // Persist explicit UI arrays for later edits
        for (i in 0 until 81) {
            uiDigits[i] = uiDigitsIn[i]
            uiConfs[i]  = uiConfsIn[i]
            uiGiven[i]  = uiGivenIn[i]
            uiSol[i]    = uiSolIn[i]
            uiAuto[i]   = uiAutoIn[i]
            uiCand[i]   = uiCandIn[i]
        }

        resultsDigits = uiDigits.copyOf()
        resultsConfidences = uiConfs.copyOf()
        lastCellReadouts = readouts9x9

        //updateSudoMessage("Hi, I’m Sudo 👋 Let’s make sure I copied your puzzle correctly.")

        // Seed overlay unresolved/editable sets from autocorrect
        lastAutoCorrectionResult?.let { ac ->
            overlayUnresolved = ac.unresolvedIndices.toMutableSet()
            overlayEditable.clear()
            overlayEditable.addAll(ac.unresolvedIndices)
            overlayEditable.addAll(ac.changedIndices)
            resultsSudokuView?.setLogicAnnotations(ac.changedIndices, overlayUnresolved.toList())
        } ?: run {
            overlayUnresolved.clear()
            overlayEditable.clear()
            resultsSudokuView?.setLogicAnnotations(emptyList(), emptyList())
        }

        // -----------------------------
        // ✅ Step-2 baseline state setup (stored in SessionStore) — keep as-is
        // -----------------------------
        val gs = buildGridStateFromOverlay()
        if (gs != null) {
            val solvability = when {
                !gs.isStructurallyValid -> SessionStore.Solvability.NONE
                gs.hasUniqueSolution -> SessionStore.Solvability.UNIQUE
                gs.hasMultipleSolutions -> SessionStore.Solvability.MULTIPLE
                else -> SessionStore.Solvability.NONE
            }

            val conflicts = gs.conflictIndices.size
            val retakeRec = when {
                conflicts >= 8 -> SessionStore.RetakeRec.STRONG
                conflicts in 4..7 -> SessionStore.RetakeRec.SOFT
                else -> SessionStore.RetakeRec.NONE
            }

            SessionStore.setStep2Baseline(
                phase = SessionStore.Step2Phase.CONFIRMING,
                mediationMode = true,
                pendingKind = null,
                pendingCellIdx = null,
                pendingDigit = null,
                lastRetakeRec = retakeRec,
                lastSolvability = solvability
            )
        } else {
            SessionStore.setStep2Baseline(
                phase = SessionStore.Step2Phase.CONFIRMING,
                mediationMode = true
            )
        }

        // Make overlay visible
        resultsRoot?.apply {
            visibility = View.VISIBLE
            alpha = 0f
            animate().alpha(1f).setDuration(180).start()
        }
        overlay.animate().alpha(0f).setDuration(120).start()
        previewView.animate().alpha(0f).setDuration(120).start()

        // -----------------------------
        // ✅ NEW: Conductor entry point (NO direct LLM call here)
        // -----------------------------
        ensureSudoStore() // defensive: guarantees store exists even if code path changes later

        val snapGs = buildGridStateFromOverlay()
        if (snapGs != null) {
            val llmGrid = buildLLMGridStateFromOverlay(snapGs)
            val gridSnap = com.contextionary.sudoku.conductor.GridSnapshot(llm = llmGrid)

            sudoStore.dispatch(com.contextionary.sudoku.conductor.Evt.GridCaptured(gridSnap))
        } else {
            sudoStore.dispatch(com.contextionary.sudoku.conductor.Evt.CameraActive)
            val msg = "I received something, but I don’t have a usable grid yet. Try retaking with the full page flat in frame."
            updateSudoMessage(msg)
            speakAssistant(msg, listenAfter = true)
        }
    }


    // Compact, consistent log lines for voice I/O
    private fun logSudoSay(text: String, speakReqId: Int? = null, replyToRowId: Int? = null) {
        // Logcat (unchanged)
        Log.i("SudoVoice", "SAY: \"${text.replace("\n"," ")}\"")

        // Telemetry JSONL (single source of truth)
        ConversationTelemetry.emitAssistantSay(
            text = text,
            source = "logSudoSay",
            speakReqId = speakReqId,
            replyToRowId = replyToRowId
            // Optional (only if you have them at this call-site):
            // mode = currentConversationMode,   // "FREE_TALK" or "GRID"
            // engine = if (isAzureConfigured()) "azure" else "android",
            // locale = localeTag
        )
    }

    private fun logAsrHeard(kind: String, text: String, confidence: Float? = null) {
        val base = "$kind: \"${text.replace("\n"," ")}\""
        if (confidence != null) {
            Log.i("SudoASR", "$base (conf=${"%.2f".format(confidence)})")
        } else {
            Log.i("SudoASR", base)
        }
    }






    private fun handleUserUtterance(userText: String) {
        lifecycleScope.launch {
            try {
                // Ensure store is ready (if you rely on lazy init)
                ensureSudoStore()

                val text = userText.trim()
                if (text.isBlank()) return@launch

                val mode = decideConversationMode()

                // Optional telemetry: keep your route event, but it no longer implies MainActivity calls LLM.
                runCatching {
                    ConversationTelemetry.emitKv(
                        "USER_UTTERANCE_ROUTE",
                        "session_id" to convoSessionId,
                        "mode" to mode.name,
                        "text_len" to text.length
                    )
                }

                when (mode) {

                    ConversationMode.FREE_TALK -> {
                        // If you already have a full FREE_TALK pipeline inside SudoStore/Conductor,
                        // you can route free talk through the store too.
                        //
                        // If not, keep your existing behavior.
                        //
                        // Recommended (single brain):
                        sudoStore.dispatch(
                            com.contextionary.sudoku.conductor.Evt.AsrFinal(
                                text = text,
                                confidence = 1.0f // or your ASR confidence if you have it
                            )
                        )
                    }

                    ConversationMode.GRID -> {
                        // ✅ IMPORTANT:
                        // Remove the old "pending confirmation interceptor" here.
                        // Pending is now handled deterministically inside SudoConductor.handlePending().
                        //
                        // Also remove:
                        // - buildGridStateFromOverlay()
                        // - buildLLMGridStateFromOverlay()
                        // - llmCoordinator.sendToLLMTools()
                        // - handleAssistantUpdate(...)
                        //
                        // MainActivity's only job is to dispatch user text into the store.

                        sudoStore.dispatch(
                            com.contextionary.sudoku.conductor.Evt.AsrFinal(
                                text = text,
                                confidence = 1.0f // or ASR confidence
                            )
                        )
                    }
                }

            } catch (t: Throwable) {
                Log.e("SudokuLLM", "handleUserUtterance failed", t)
            }
        }
    }





    private fun buildLLMGridStateFromOverlay(
        state: com.contextionary.sudoku.logic.GridState
    ): LLMGridState {

        val correctedGrid = state.digits.copyOf()

        SessionStore.ensureTruthInitializedFromCanonicalIfPossible()
        val truthSnap = SessionStore.truthSnapshotOrNull()

        fun safeBool81(x: BooleanArray?, fallback: BooleanArray): BooleanArray {
            return if (x != null && x.size == 81) x.copyOf() else fallback.copyOf()
        }

        fun safeInt81(x: IntArray?, fallback: IntArray): IntArray {
            return if (x != null && x.size == 81) x.copyOf() else fallback.copyOf()
        }

        val truthGiven = safeBool81(truthSnap?.isGiven, uiGiven)
        val truthSol = safeBool81(truthSnap?.isSolution, uiSol)
        val truthCand = safeInt81(truthSnap?.candidateMask, uiCand)

        // ✅ Confirmed indices (handled already; should not be re-asked)
        fun isConfirmed(idx: Int): Boolean = (idx in 0..80 && uiConfirmed[idx])

        // ✅ NEW: explicit confirmed list for LLMGridState (facts for coordinator prompt)
        val confirmedCells = (0 until 81)
            .asSequence()
            .filter { uiConfirmed[it] }
            .toList()
            .sorted()

        val unresolvedCells = state.unresolvedIndices
            .toSet()
            .toList()
            .sorted()
            .filterNot { isConfirmed(it) }  // ✅ keeps "next-check" from looping

        val changedCells = state.changedByLogic.toSet().toList().sorted()
        val conflictCells = state.conflictIndices.toSet().toList().sorted()

        val lowConfidenceCells = state.cells
            .asSequence()
            .filter { it.isLowConfidence }
            .map { it.index }
            .toSet()
            .toList()
            .sorted()

        val manuallyCorrectedCells = (0 until 81)
            .asSequence()
            .filter { uiManual[it] }
            .toList()

        val manualEdits = manualEditLog
            .sortedBy { it.seq }
            .toList()

        // Deduced solution from GIVENS ONLY
        val givensOnly = IntArray(81) { idx -> if (truthGiven[idx]) correctedGrid[idx].coerceIn(0, 9) else 0 }
        val solveRes = com.contextionary.sudoku.logic.DeterministicSudokuSolver
            .solveCountCapped(givensOnly, maxCount = 2)
        val deducedSolution = if (solveRes.solutionCount == 1) solveRes.solutionGrid else null

        // Mismatch detection (only meaningful when unique)
        val mismatchCellsRaw = mutableListOf<Int>()
        if (deducedSolution != null) {
            for (idx in 0 until 81) {
                if (!truthSol[idx]) continue
                val userVal = correctedGrid[idx].coerceIn(0, 9)
                if (userVal == 0) continue
                val expected = deducedSolution[idx].coerceIn(0, 9)
                if (expected != 0 && userVal != expected) mismatchCellsRaw += idx
            }
        }

        // ✅ filter mismatch by confirmed (prevents asking same mismatch cell again)
        val mismatchCellsSorted = mismatchCellsRaw
            .distinct()
            .sorted()
            .filterNot { isConfirmed(it) }

        val mismatchDetails = mutableListOf<String>()
        if (deducedSolution != null) {
            for (idx in mismatchCellsSorted) {
                val r = (idx / 9) + 1
                val c = (idx % 9) + 1
                val userVal = correctedGrid[idx].coerceIn(0, 9)
                val expected = deducedSolution[idx].coerceIn(0, 9)
                mismatchDetails += "r${r}c${c} expected=$expected user=$userVal"
            }
        }

        val solvability = when {
            state.hasUniqueSolution -> "unique"
            state.hasMultipleSolutions -> "multiple"
            else -> "none"
        }

        val isStructurallyValid = state.isStructurallyValid
        val unresolvedCount = unresolvedCells.size

        // Retake policy aligned with "Sudo solves only unique"
        val retakeRecommendation = when {
            conflictCells.size >= 8 -> "strong"
            conflictCells.size in 4..7 -> "soft"
            solvability == "multiple" -> "soft"
            unresolvedCells.size > 6 -> "soft"
            else -> "none"
        }

        // Severity: treat multiple as serious (not acceptable for solving)
        val severity = when {
            solvability == "none" -> "serious"
            solvability == "multiple" -> "serious"
            conflictCells.isNotEmpty() -> "serious"
            unresolvedCells.isNotEmpty() -> "serious"
            lowConfidenceCells.isNotEmpty() -> "mild"
            changedCells.isNotEmpty() -> "mild"
            else -> "ok"
        }

        return LLMGridState(
            correctedGrid = correctedGrid,

            truthIsGiven = truthGiven,
            truthIsSolution = truthSol,
            candidateMask81 = truthCand,

            unresolvedCells = unresolvedCells,
            changedCells = changedCells,
            conflictCells = conflictCells,
            lowConfidenceCells = lowConfidenceCells,

            manuallyCorrectedCells = manuallyCorrectedCells,
            manualEdits = manualEdits,

            deducedSolutionGrid = deducedSolution,
            deducedSolutionCountCapped = solveRes.solutionCount,
            mismatchCells = mismatchCellsSorted,
            mismatchDetails = mismatchDetails,

            // ✅ NEW: pass confirmations through to coordinator prompt
            confirmedCells = confirmedCells,

            solvability = solvability,
            isStructurallyValid = isStructurallyValid,
            unresolvedCount = unresolvedCount,
            severity = severity,
            retakeRecommendation = retakeRecommendation
        )
    }



    private fun applyTruthReclassification(idx: Int, kind: String, source: String) {
        val ok = SessionStore.reclassifyCell(idx, kind)
        if (!ok) return

        // Reflect truth back into UI flags for rendering
        val snap = SessionStore.truthSnapshotOrNull() ?: return
        for (i in 0 until 81) {
            uiGiven[i] = snap.isGiven[i]
            uiSol[i] = snap.isSolution[i]
        }

        // Re-render
        resultsSudokuView?.setUiData(
            displayDigits = uiDigits,
            displayConfs = uiConfs,
            shownIsGiven = uiGiven,
            shownIsSolution = uiSol,
            candidatesMask = uiCand,
            shownIsAutoCorrected = uiAuto
        )

        // Send fresh grid snapshot to conductor
        val gs = buildGridStateFromOverlay() ?: return
        val llmGrid = buildLLMGridStateFromOverlay(gs)
        sudoStore.dispatch(com.contextionary.sudoku.conductor.Evt.GridSnapshotUpdated(
            com.contextionary.sudoku.conductor.GridSnapshot(llm = llmGrid)
        ))
    }

    private fun applyCandidatesSet(idx: Int, mask: Int, source: String) {
        val ok = SessionStore.setCandidates(idx, mask)
        if (!ok) return

        val snap = SessionStore.truthSnapshotOrNull() ?: return
        for (i in 0 until 81) {
            uiCand[i] = snap.candidateMask[i]
        }

        resultsSudokuView?.setUiData(
            displayDigits = uiDigits,
            displayConfs = uiConfs,
            shownIsGiven = uiGiven,
            shownIsSolution = uiSol,
            candidatesMask = uiCand,
            shownIsAutoCorrected = uiAuto
        )

        val gs = buildGridStateFromOverlay() ?: return
        val llmGrid = buildLLMGridStateFromOverlay(gs)
        sudoStore.dispatch(com.contextionary.sudoku.conductor.Evt.GridSnapshotUpdated(
            com.contextionary.sudoku.conductor.GridSnapshot(llm = llmGrid)
        ))
    }

    private fun applyCandidatesToggle(idx: Int, digit: Int, source: String) {
        val ok = SessionStore.toggleCandidate(idx, digit)
        if (!ok) return
        val snap = SessionStore.truthSnapshotOrNull() ?: return

        for (i in 0 until 81) {
            uiCand[i] = snap.candidateMask[i]
        }

        resultsSudokuView?.setUiData(
            displayDigits = uiDigits,
            displayConfs = uiConfs,
            shownIsGiven = uiGiven,
            shownIsSolution = uiSol,
            candidatesMask = uiCand,
            shownIsAutoCorrected = uiAuto
        )

        val gs = buildGridStateFromOverlay() ?: return
        val llmGrid = buildLLMGridStateFromOverlay(gs)
        sudoStore.dispatch(com.contextionary.sudoku.conductor.Evt.GridSnapshotUpdated(
            com.contextionary.sudoku.conductor.GridSnapshot(llm = llmGrid)
        ))
    }



    private fun extractDigit1to9(text: String): Int? {
        val t = text.lowercase()

        // A) Prefer explicit "to X" / "equals X" / "is X" / "= X"
        Regex("""\b(?:to|equals|equal|is|=)\s*([1-9])\b""")
            .find(t)?.groupValues?.getOrNull(1)?.toIntOrNull()
            ?.let { return it }

        // B) Prefer explicit "to six" etc
        val wordToDigit = mapOf(
            "one" to 1,
            "two" to 2,
            "three" to 3,
            "four" to 4,
            "for" to 4,     // common ASR
            "five" to 5,
            "six" to 6,
            "sex" to 6,     // common ASR
            "seven" to 7,
            "eight" to 8,
            "ate" to 8,     // common ASR
            "nine" to 9
        )
        Regex("""\b(?:to|equals|equal|is|=)\s*(one|two|three|four|for|five|six|sex|seven|eight|ate|nine)\b""")
            .find(t)?.groupValues?.getOrNull(1)
            ?.let { w -> wordToDigit[w]?.let { return it } }

        // C) Otherwise, take the LAST standalone numeric digit 1..9 (avoid grabbing "r1c1" because no word-boundary)
        Regex("""\b([1-9])\b""").findAll(t).toList().lastOrNull()
            ?.groupValues?.getOrNull(1)?.toIntOrNull()
            ?.let { return it }

        // D) Otherwise, take the LAST number-word occurrence
        val hits = wordToDigit.entries
            .filter { (k, _) -> Regex("""\b$k\b""").containsMatchIn(t) }
        hits.lastOrNull()?.value?.let { return it }

        return null
    }



    private fun applyManualDigitEdit(idx: Int, newDigit: Int) {
        if (idx !in 0..80) return
        if (newDigit !in 1..9) return
        if (resultsDigits == null || resultsConfidences == null || resultsSudokuView == null) return

        val oldDigit = uiDigits[idx]
        if (oldDigit == newDigit) {
            // No-op is safe
            overlayUnresolved.remove(idx)
            resultsSudokuView?.stopConfirmationPulse()
            return
        }

        // 1) Update canonical UI arrays (what is displayed)
        uiDigits[idx] = newDigit
        uiConfs[idx] = 1.0f

        // Bold is driven by corrected flags, not "given/solution" booleans
        uiGiven[idx] = false
        uiSol[idx] = false
        uiCand[idx] = 0

        // 2) Mark manual provenance
        uiManual[idx] = true

        // 3) Append manual edit history for the LLM (epoch time; stable)
        manualEditSeqGlobal += 1
        manualEditLog.add(
            com.contextionary.sudoku.logic.LLMCellEditEvent(
                seq = manualEditSeqGlobal,
                index = idx,
                cellLabel = "r${(idx / 9) + 1}c${(idx % 9) + 1}",
                fromDigit = oldDigit,
                toDigit = newDigit,
                whenEpochMs = System.currentTimeMillis(),
                source = "manual"
            )
        )

        // 4) Keep resultsDigits/resultsConfidences consistent (canonical for solver + GridState)
        System.arraycopy(uiDigits, 0, resultsDigits!!, 0, 81)
        System.arraycopy(uiConfs, 0, resultsConfidences!!, 0, 81)

        // 5) Remove from unresolved
        overlayUnresolved.remove(idx)

        // 6) Repaint with boldCorrected = auto OR manual
        val boldCorrected = BooleanArray(81) { i -> uiAuto[i] || uiManual[i] }

        resultsSudokuView?.setUiData(
            displayDigits = uiDigits,
            displayConfs = uiConfs,
            shownIsGiven = uiGiven,
            shownIsSolution = uiSol,
            candidatesMask = uiCand,
            // NOTE: View param is misnamed; you're using it as "corrected => bold"
            shownIsAutoCorrected = boldCorrected
        )

        // 7) Re-apply logic annotations (cyan=auto-changed, red=unresolved)
        val changedCellsIdx = lastAutoCorrectionResult?.changedIndices ?: emptyList()
        resultsSudokuView?.setLogicAnnotations(
            changed = changedCellsIdx,
            unresolved = overlayUnresolved.toList()
        )

        resultsSudokuView?.stopConfirmationPulse()

        // 8) Optional: notify GridConversationCoordinator + log event (your existing pattern)
        buildGridStateFromOverlay()?.let { state ->
            val editEvent = com.contextionary.sudoku.logic.GridEditEvent(
                seq = manualEditSeqGlobal,
                cellIndex = idx,
                row = idx / 9,
                col = idx % 9,
                oldDigit = oldDigit,
                newDigit = newDigit,
                timestampMs = android.os.SystemClock.elapsedRealtime()
            )
            gridConversationCoordinator.onManualEditApplied(state, editEvent)
        }
    }




    private fun updateProfileFrom(userText: String) {
        lifecycleScope.launch {
            runCatching {
                val delta = llmCoordinator.extractProfileClues(userText)
                val current = UserProfileStore.load(this@MainActivity)
                UserProfileStore.mergeAndSave(this@MainActivity, current, delta)
            }.onFailure {
                android.util.Log.w("SudoProfile", "Profile extraction failed (non-fatal): ${it.message}")
            }
        }
    }


    private fun RectF.toAndroidRect(): Rect =
        Rect(left.toInt(), top.toInt(), right.toInt(), bottom.toInt())

    // Flat → 9×9 helpers
    private fun to9x9Int(flat: IntArray): Array<IntArray> {
        require(flat.size == 81) { "Expected 81 ints" }
        return Array(9) { r -> IntArray(9) { c -> flat[r * 9 + c] } }
    }

    private fun to9x9Float(flat: FloatArray): Array<FloatArray> {
        require(flat.size == 81) { "Expected 81 floats" }
        return Array(9) { r -> FloatArray(9) { c -> flat[r * 9 + c] } }
    }





    // Drop ALL other overloads. Keep ONLY this one.
    // Drop ALL other overloads. Keep ONLY this one.
    private fun saveRoiForParity(
        fullFrame: Bitmap,
        detectorRoi: android.graphics.Rect,
        caseId: String,
        options: Rectifier.Options? = null   // ← default provided so callers may omit it
    ): java.io.File {
        val opts = options ?: Rectifier.Options()

        val root = java.io.File(filesDir, "runs/grid_rectification/$caseId")
        root.mkdirs()

        val left   = detectorRoi.left.coerceAtLeast(0)
        val top    = detectorRoi.top.coerceAtLeast(0)
        val right  = detectorRoi.right.coerceAtMost(fullFrame.width)
        val bottom = detectorRoi.bottom.coerceAtMost(fullFrame.height)
        val w = (right - left).coerceAtLeast(1)
        val h = (bottom - top).coerceAtLeast(1)
        val roiBmp = Bitmap.createBitmap(fullFrame, left, top, w, h)

        java.io.File(root, "roi.png").outputStream().use { out ->
            roiBmp.compress(Bitmap.CompressFormat.PNG, 100, out)
        }

        val manifest = """
      {
        "tile_size": ${opts.tileSize},
        "shrink": ${opts.shrink},
        "precompress": ${opts.precompress},
        "target_kb": ${opts.targetKb},
        "max_side": ${opts.maxSide},
        "min_jpeg_quality": ${opts.minJpegQuality},
        "robust": ${opts.robust},
        "roi": {"x0": $left, "y0": $top, "x1": $right, "y1": $bottom},
        "source": {"w": ${fullFrame.width}, "h": ${fullFrame.height}},
        "engine_hint": "android",
        "case_id": "$caseId"
      }
    """.trimIndent()
        java.io.File(root, "manifest.json").writeText(manifest, Charsets.UTF_8)

        return root
    }


    // --- Debug I/O helpers ------------------------------------------------------



    // ===== Rectify debug saving =====


    private fun wipeDir(dir: java.io.File) {
        if (!dir.exists()) return
        dir.listFiles()?.forEach {
            if (it.isDirectory) wipeDir(it) else runCatching { it.delete() }
        }
    }



    // ===== Geometry helpers (PointF versions) =====


    private fun updateSudoMessage(text: String?) {
        runOnUiThread {
            val tv = sudoMessageTextView ?: return@runOnUiThread
            if (text.isNullOrBlank()) {
                tv.text = ""
                tv.visibility = View.INVISIBLE
            } else {
                tv.text = text
                tv.visibility = View.VISIBLE
            }
        }
    }



    /**
     * Handle an AskUserConfirmation action from the LLM:
     *  - highlight the target cell with a pulsing border
     *  - show the bottom digit picker strip
     *  - select that cell for the next digit tap
     */
    private fun handleAskUserConfirmation(cellIndex: Int) {
        var idx = cellIndex
        if (idx !in 0..80) return
        if (resultsDigits == null || resultsConfidences == null || resultsSudokuView == null) return

        // If the chosen cell is NOT actually doubtful, redirect to a better candidate:
        val digit = resultsDigits!![idx]
        val conf  = resultsConfidences!![idx]
        val isChangedByLogic = lastAutoCorrectionResult?.changedIndices?.contains(idx) == true
        val isLowConf = (digit != 0 && conf < SudokuConfidence.THRESH_HIGH)

        if (!isChangedByLogic && !isLowConf) {
            val betterIdx =
                (0 until 81).firstOrNull { i ->
                    val d = resultsDigits!![i]
                    val c = resultsConfidences!![i]
                    (d != 0 && c < SudokuConfidence.THRESH_HIGH)
                } ?: lastAutoCorrectionResult?.changedIndices?.firstOrNull()

            if (betterIdx != null) idx = betterIdx
        }

        overlayEditable.add(idx)
        selectedOverlayIdx = idx

        resultsSudokuView?.startConfirmationPulse(idx)
        digitPickerRow?.visibility = View.VISIBLE
    }





    /** Ensure a row contains exactly 10 points by linear interpolation along X. */
    private fun interpolateRowToTen(sortedRow: List<PointF>): Array<PointF> {
        val out = Array(10) { PointF() }
        if (sortedRow.isEmpty()) {
            // Fallback: make a dummy row—caller will likely fail geometry later.
            for (i in 0 until 10) out[i] = PointF(i.toFloat(), 0f)
            return out
        }
        val left = sortedRow.first().x
        val right = sortedRow.last().x
        if (right <= left + 1e-3f) {
            // Degenerate; collapse to left
            for (i in 0 until 10) out[i] = PointF(left, sortedRow.first().y)
            return out
        }

        // Build a piecewise-linear map of X→Y using the existing points,
        // then sample it at 10 evenly spaced Xs between [left, right].
        val xs = sortedRow.map { it.x }.toFloatArray()
        val ys = sortedRow.map { it.y }.toFloatArray()

        fun yAt(xq: Float): Float {
            // find bracketing segment
            var i = xs.indexOfLast { it <= xq }
            if (i < 0) return ys.first()
            if (i >= xs.size - 1) return ys.last()
            val x0 = xs[i]; val x1 = xs[i + 1]
            val y0 = ys[i]; val y1 = ys[i + 1]
            val t = ((xq - x0) / (x1 - x0)).coerceIn(0f, 1f)
            return y0 + t * (y1 - y0)
        }

        for (i in 0 until 10) {
            val xq = left + (right - left) * (i / 9f)
            out[i] = PointF(xq, yAt(xq))
        }
        return out
    }


    private data class Quadruple(val s: Float, val dw: Float, val dh: Float, val offX: Float, val offY: Float)



    // Check if a rect "touches" the border (stroke) of a square, within tolerance.
    // Determine if the detection box touches the guard border within a tolerance (source space).
    private fun touchesBorder(r: RectF, border: RectF, tolPx: Float): Boolean {
        fun overlap1D(a1: Float, a2: Float, b1: Float, b2: Float): Boolean {
            val lo = max(a1, b1)
            val hi = min(a2, b2)
            return hi >= lo
        }
        // Left side
        val touchLeft = (kotlin.math.abs(r.right - border.left) <= tolPx ||
                kotlin.math.abs(r.left - border.left) <= tolPx) &&
                overlap1D(r.top, r.bottom, border.top, border.bottom)

        // Right side
        val touchRight = (kotlin.math.abs(r.left - border.right) <= tolPx ||
                kotlin.math.abs(r.right - border.right) <= tolPx) &&
                overlap1D(r.top, r.bottom, border.top, border.bottom)

        // Top side
        val touchTop = (kotlin.math.abs(r.bottom - border.top) <= tolPx ||
                kotlin.math.abs(r.top - border.top) <= tolPx) &&
                overlap1D(r.left, r.right, border.left, border.right)

        // Bottom side
        val touchBottom = (kotlin.math.abs(r.top - border.bottom) <= tolPx ||
                kotlin.math.abs(r.bottom - border.bottom) <= tolPx) &&
                overlap1D(r.left, r.right, border.left, border.right)

        return touchLeft || touchRight || touchBottom || touchTop
    }
}

// RGBA8888 -> Bitmap (unchanged)
private fun ImageProxy.toBitmapRGBA(): Bitmap? {
    val plane = planes.firstOrNull() ?: return null
    val w = width
    val h = height
    val rowStride = plane.rowStride
    val pixelStride = plane.pixelStride
    if (pixelStride != 4) {
        Log.w("MainActivity", "Unexpected pixelStride=$pixelStride for RGBA_8888")
        return null
    }

    val needed = w * h * 4
    val contiguous = (rowStride == w * 4)

    val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
    val buffer: java.nio.ByteBuffer = plane.buffer

    return try {
        if (contiguous) {
            buffer.rewind()
            val slice = buffer.duplicate()
            slice.rewind()
            val safeLimit = min(slice.capacity(), needed)
            slice.limit(safeLimit)
            bmp.copyPixelsFromBuffer(slice)
        } else {
            val row = ByteArray(w * 4)
            val dst = java.nio.ByteBuffer.allocateDirect(needed)
            for (y in 0 until h) {
                buffer.position(y * rowStride)
                buffer.get(row, 0, row.size)
                dst.put(row)
            }
            dst.rewind()
            bmp.copyPixelsFromBuffer(dst)
        }
        bmp
    } catch (t: Throwable) {
        Log.e("MainActivity", "toBitmapRGBA() failed", t)
        null
    }
}