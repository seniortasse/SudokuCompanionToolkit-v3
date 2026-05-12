package com.contextionary.sudoku.conductor.solving

import org.json.JSONArray
import org.json.JSONObject

/**
 * SolveStepV2 is the single source of truth for the solving engine step contract.
 *
 * Android receives an envelope from Python:
 *   { ok, status, step?, error? }
 * When status == "ok", "step" MUST be a SolveStepV2 object.
 */

// ----------------------------
// Envelope
// ----------------------------

data class SolveStepV2Envelope(
    val ok: Boolean,
    val status: String,
    val step: SolveStepV2? = null,
    val error: SolveStepError? = null,
    val raw: JSONObject
)

data class SolveStepError(
    val code: String,
    val msg: String? = null
)

// ----------------------------
// SolveStepV2 (top-level)
// ----------------------------

data class SolveStepV2(
    val schemaVersion: String,
    val engine: EngineInfo,
    val ids: StepIds,
    val technique: TechniqueInfo,
    val grids: Grids? = null,
    val target: Target,
    val proof: Proof,
    val hintLadder: List<HintFrame> = emptyList(),
    val overlayFrames: List<OverlayFrameV2> = emptyList(),
    val teaching: Teaching = Teaching(),
    val engineNative: JSONObject? = null,
    val raw: JSONObject
)

data class EngineInfo(
    val name: String,
    val engineVersion: String? = null,
    val useCleanupMethod: Boolean = true,
    val includeMagicTechnique: Boolean = false
)

data class StepIds(
    val gridHash12Before: String,
    val gridHash12After: String,
    val stepId: String,
    val stepIndex: Int = 1
)

data class TechniqueDefinition(
    val what: String? = null,
    val why: String? = null,
    val whenText: String? = null, // JSON key: "when"
    val how: String? = null
)

data class TechniqueComments(
    val synonyms: List<String> = emptyList(),
    val notes: String? = null,
    val interestingFacts: List<String> = emptyList()
)

data class TechniqueInfo(
    val techniqueId: String,
    val techniqueName: String,
    val priorityRank: Int,
    val difficultyWeight: Int,
    val family: String? = null,

    // ✅ richer fields (all nullable; doesn’t break older JSON)
    val appName: String? = null,
    val realName: String? = null,
    val familyDescription: String? = null,
    val difficultyLevel: String? = null,
    val isBase: Boolean? = null,
    val definition: TechniqueDefinition? = null,
    val comments: TechniqueComments? = null
)

data class Grids(
    val grid81Before: String,
    val grid81After: String
)

data class Target(
    val kind: String,
    val cell: TargetCell? = null,
    val digit: Int? = null
)

data class TargetCell(
    val r: Int,
    val c: Int,
    val cellIndex: Int
)

data class Proof(
    val placements: List<Placement> = emptyList(),
    val eliminations: List<Elimination> = emptyList(),
    val housesInvolved: List<HouseRefV2> = emptyList(),
    val houseSummaries: List<HouseSummaryV2> = emptyList(),

    // Phase 3: evidence-first hint ladder
    val hintLadderV1: List<HintEvidenceV2> = emptyList(),

    val candidates: Candidates = Candidates(),
    val applications: List<ApplicationV2> = emptyList(),
    val raw: JSONObject
)

data class HintEvidenceV2(
    val index: Int,
    val kind: String,
    val overlayFrameId: String?,
    val evidence: JSONObject,
    val raw: JSONObject
)

data class HouseSummaryV2(
    val kind: String,                 // "row" | "col" | "box"
    val index1to9: Int,               // 1..9
    val presentDigits: List<Int> = emptyList(),
    val missingDigits: List<Int> = emptyList(),
    val emptyCells: List<TargetCell> = emptyList(),
    val raw: JSONObject
)

data class Placement(
    val cellIndex: Int,
    val r: Int,
    val c: Int,
    val digit: Int,
    val source: String? = null
)

data class Elimination(
    val cellIndex: Int,
    val r: Int,
    val c: Int,
    val digit: Int,
    val reasonCode: String? = null
)

data class HouseRefV2(
    val type: String,
    val index1to9: Int
)

data class Candidates(
    val snapshotBefore: Map<String, Int> = emptyMap(),
    val snapshotAfter: Map<String, Int> = emptyMap(),
    val relevantCells: List<Int> = emptyList()
)

data class SupportV2(
    val patternCells: List<Int> = emptyList(), // cells defining the pattern
    val peerCells: List<Int> = emptyList()     // impacted peers / houses touched
)

data class EffectsV2(
    val eliminations: List<Elimination> = emptyList()
)

data class ApplicationV2(
    val applicationId: String,
    val techniqueId: String,
    val nameApplication: String,
    val kind: String,
    val digits: List<Int> = emptyList(),
    val houses: List<HouseRefV2> = emptyList(),

    // ✅ typed for milestone 1/4/5 rendering
    val supportV2: SupportV2? = null,
    val effectsV2: EffectsV2? = null,

    // keep raw JSON too (for forward-compat / debugging)
    val pattern: JSONObject? = null,
    val effects: JSONObject? = null,
    val support: JSONObject? = null,

    val engineNative: JSONObject? = null,
    val raw: JSONObject
)

data class HintFrame(
    val frameId: String,
    val text: String,
    val raw: JSONObject
)

data class OverlayFrameV2(
    val frameId: String,
    val style: String,
    val highlights: List<OverlayHighlight> = emptyList(),
    val raw: JSONObject
)

data class OverlayHighlight(
    val kind: String, // "cell" | "house" | "link"
    val cellIndex: Int? = null,
    val role: String? = null,
    val digit: Int? = null,
    val house: HouseRefV2? = null,

    // Phase 3: link highlight (for fish/wings/chains)
    val fromCellIndex: Int? = null,
    val toCellIndex: Int? = null,

    val raw: JSONObject
)

data class Teaching(
    val recognition: List<String> = emptyList(),
    val application: List<String> = emptyList(),
    val pitfalls: List<String> = emptyList(),
    val glossaryKeys: List<String> = emptyList()
)

// ----------------------------
// Parser
// ----------------------------

object SolveStepV2Parser {

    fun parseEnvelope(json: String): SolveStepV2Envelope? {
        val root = runCatching { JSONObject(json) }.getOrNull() ?: return null

        val ok = root.optBoolean("ok", false)
        val status = root.optString("status").ifBlank { "unknown" }

        val errObj = root.optJSONObject("error")
        val err = errObj?.let {
            SolveStepError(
                code = it.optString("code").ifBlank { "unknown" },
                msg = it.optString("msg").ifBlank { null }
            )
        }

        val stepObj = root.optJSONObject("step")
        val step = stepObj?.let { parseStep(it) }

        return SolveStepV2Envelope(
            ok = ok,
            status = status,
            step = step,
            error = err,
            raw = root
        )
    }

    fun parseStep(step: JSONObject): SolveStepV2? {
        val schemaVersion = step.optString("schema_version").ifBlank { "" }
        if (schemaVersion != "solve_step_v2") return null

        val engineObj = step.optJSONObject("engine") ?: JSONObject()
        val engine = EngineInfo(
            name = engineObj.optString("name").ifBlank { "unknown" },
            engineVersion = engineObj.optString("engine_version").ifBlank { null },
            useCleanupMethod = engineObj.optBoolean("use_cleanup_method", true),
            includeMagicTechnique = engineObj.optBoolean("include_magic_technique", false)
        )

        val idsObj = step.optJSONObject("ids") ?: JSONObject()
        val ids = StepIds(
            gridHash12Before = idsObj.optString("grid_hash12_before").ifBlank { "unknown" },
            gridHash12After = idsObj.optString("grid_hash12_after").ifBlank { "unknown" },
            stepId = idsObj.optString("step_id").ifBlank { "unknown" },
            stepIndex = idsObj.optInt("step_index", 1)
        )

        val techObj = step.optJSONObject("technique") ?: JSONObject()

        val defObj = techObj.optJSONObject("definition") ?: JSONObject()
        val definition = TechniqueDefinition(
            what = defObj.optString("what").ifBlank { null },
            why = defObj.optString("why").ifBlank { null },
            whenText = defObj.optString("when").ifBlank { null },
            how = defObj.optString("how").ifBlank { null }
        )

        val commentsObj = techObj.optJSONObject("comments") ?: JSONObject()
        val synonymsArr = commentsObj.optJSONArray("synonyms") ?: JSONArray()
        val interestingArr = commentsObj.optJSONArray("interesting_facts") ?: JSONArray()

        val synonyms = (0 until synonymsArr.length()).mapNotNull { i ->
            synonymsArr.optString(i).takeIf { it.isNotBlank() }
        }

        val interestingFacts = (0 until interestingArr.length()).mapNotNull { i ->
            interestingArr.optString(i).takeIf { it.isNotBlank() }
        }

        val comments = TechniqueComments(
            synonyms = synonyms,
            notes = commentsObj.optString("notes").ifBlank { null },
            interestingFacts = interestingFacts
        )

        val technique = TechniqueInfo(
            techniqueId = techObj.optString("technique_id").ifBlank { techObj.optString("id").ifBlank { "unknown" } },
            techniqueName = techObj.optString("technique_name").ifBlank {
                techObj.optString("technique_id").ifBlank { techObj.optString("id").ifBlank { "unknown" } }
            },
            priorityRank = techObj.optInt("priority_rank", techObj.optInt("priorityRank", 0)),
            difficultyWeight = techObj.optInt("difficulty_weight", techObj.optInt("difficultyWeight", 0)),
            family = techObj.optString("family").ifBlank { null },

            appName = techObj.optString("app_name").ifBlank { techObj.optString("appName").ifBlank { null } },
            realName = techObj.optString("real_name").ifBlank { techObj.optString("realName").ifBlank { null } },
            familyDescription = techObj.optString("family_description").ifBlank { techObj.optString("familyDescription").ifBlank { null } },
            difficultyLevel = techObj.optString("difficulty_level").ifBlank { techObj.optString("difficultyLevel").ifBlank { null } },

            isBase = if (techObj.has("is_base") || techObj.has("isBase")) {
                if (techObj.has("is_base")) techObj.optBoolean("is_base") else techObj.optBoolean("isBase")
            } else null,

            definition = definition,
            comments = comments
        )

        val gridsObj = step.optJSONObject("grids")
        val grids = gridsObj?.let {
            val b = it.optString("grid81_before")
            val a = it.optString("grid81_after")
            if (b.length == 81 && a.length == 81) Grids(b, a) else null
        }

        val targetObj = step.optJSONObject("target") ?: JSONObject()
        val cellObj = targetObj.optJSONObject("cell")
        val target = Target(
            kind = targetObj.optString("kind").ifBlank { "noop" },
            cell = cellObj?.let {
                val r = optIntAny(it, "r", default = -1)
                val c = optIntAny(it, "c", default = -1)
                val ci = optIntAny(it, "cell_index", "cellIndex", default = -1)
                TargetCell(r = r, c = c, cellIndex = ci)
                    .takeIf { tc -> tc.cellIndex in 0..80 && tc.r in 1..9 && tc.c in 1..9 }
            },
            digit = optIntAny(targetObj, "digit", default = -1).takeIf { it in 1..9 }
        )

        val proofObj = step.optJSONObject("proof") ?: JSONObject()
        val proof = Proof(
            placements = parsePlacements(proofObj.optJSONArray("placements")),
            eliminations = parseEliminations(proofObj.optJSONArray("eliminations")),
            housesInvolved = parseHouses(proofObj.optJSONArray("houses_involved")),
            houseSummaries = parseHouseSummaries(proofObj.optJSONArray("house_summaries")),
            hintLadderV1 = parseHintLadderV1(proofObj.optJSONArray("hint_ladder_v1")),
            candidates = parseCandidates(proofObj.optJSONObject("candidates")),
            applications = parseApplications(proofObj.optJSONArray("applications")),
            raw = proofObj
        )

        val hintLadder = parseHintLadder(step.optJSONArray("hint_ladder"))
        val overlayFrames = parseOverlayFrames(step.optJSONArray("overlay_frames"))
        val teaching = parseTeaching(step.optJSONObject("teaching"))

        val engineNative = step.optJSONObject("engine_native")

        return SolveStepV2(
            schemaVersion = schemaVersion,
            engine = engine,
            ids = ids,
            technique = technique,
            grids = grids,
            target = target,
            proof = proof,
            hintLadder = hintLadder,
            overlayFrames = overlayFrames,
            teaching = teaching,
            engineNative = engineNative,
            raw = step
        )
    }


    private fun parseHintLadderV1(arr: JSONArray?): List<HintEvidenceV2> {
        if (arr == null) return emptyList()
        val out = ArrayList<HintEvidenceV2>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue
            val idx = optIntAny(o, "index", default = i)
            val kind = o.optString("kind")
            if (kind.isBlank()) continue
            val overlay = o.optString("overlay_frame_id").takeIf { it.isNotBlank() }
            val evidence = o.optJSONObject("evidence") ?: JSONObject()
            out += HintEvidenceV2(
                index = idx,
                kind = kind,
                overlayFrameId = overlay,
                evidence = evidence,
                raw = o
            )
        }
        return out.sortedBy { it.index }
    }

    private fun parsePlacements(arr: JSONArray?): List<Placement> {
        if (arr == null) return emptyList()
        val out = ArrayList<Placement>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue
            val ci = optIntAny(o, "cell_index", "cellIndex", default = -1)
            val r = optIntAny(o, "r", default = -1)
            val c = optIntAny(o, "c", default = -1)
            val d = optIntAny(o, "digit", default = -1)
            if (ci !in 0..80 || r !in 1..9 || c !in 1..9 || d !in 1..9) continue
            out += Placement(
                cellIndex = ci,
                r = r,
                c = c,
                digit = d,
                source = optStringAny(o, "source").ifBlank { null }
            )
        }
        return out
    }

    private fun parseEliminations(arr: JSONArray?): List<Elimination> {
        if (arr == null) return emptyList()
        val out = ArrayList<Elimination>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue
            val ci = optIntAny(o, "cell_index", "cellIndex", default = -1)
            val r = optIntAny(o, "r", default = -1)
            val c = optIntAny(o, "c", default = -1)
            val d = optIntAny(o, "digit", default = -1)
            if (ci !in 0..80 || r !in 1..9 || c !in 1..9 || d !in 1..9) continue
            out += Elimination(
                cellIndex = ci,
                r = r,
                c = c,
                digit = d,
                reasonCode = optStringAny(o, "reason_code", "reasonCode").ifBlank { null }
            )
        }
        return out
    }

    private fun parseHouses(arr: JSONArray?): List<HouseRefV2> {
        if (arr == null) return emptyList()
        val out = ArrayList<HouseRefV2>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue
            val t = o.optString("type")
            if (t.isBlank()) continue
            val idx = o.optInt("index1to9", -1)
            if (idx !in 1..9) continue
            out += HouseRefV2(type = t, index1to9 = idx)
        }
        return out
    }


    private fun parseHouseSummaries(arr: JSONArray?): List<HouseSummaryV2> {
        if (arr == null) return emptyList()
        val out = ArrayList<HouseSummaryV2>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue
            val kind = o.optString("kind")
            if (kind.isBlank()) continue
            val idx = optIntAny(o, "index1to9", "index_1to9", default = -1)
            if (idx !in 1..9) continue

            val present = mutableListOf<Int>()
            val presentArr = o.optJSONArray("present_digits")
            if (presentArr != null) {
                for (j in 0 until presentArr.length()) {
                    val d = presentArr.optInt(j, -1)
                    if (d in 1..9) present.add(d)
                }
            }

            val missing = mutableListOf<Int>()
            val missingArr = o.optJSONArray("missing_digits")
            if (missingArr != null) {
                for (j in 0 until missingArr.length()) {
                    val d = missingArr.optInt(j, -1)
                    if (d in 1..9) missing.add(d)
                }
            }

            val empties = mutableListOf<TargetCell>()
            val emptyArr = o.optJSONArray("empty_cells")
            if (emptyArr != null) {
                for (j in 0 until emptyArr.length()) {
                    val cObj = emptyArr.optJSONObject(j) ?: continue
                    val r = optIntAny(cObj, "r", default = -1)
                    val cc = optIntAny(cObj, "c", default = -1)
                    val ci = optIntAny(cObj, "cellIndex", "cell_index", default = -1)
                    if (ci in 0..80 && r in 1..9 && cc in 1..9) {
                        empties.add(TargetCell(r = r, c = cc, cellIndex = ci))
                    }
                }
            }

            out += HouseSummaryV2(
                kind = kind,
                index1to9 = idx,
                presentDigits = present.sorted(),
                missingDigits = missing,
                emptyCells = empties,
                raw = o
            )
        }
        return out
    }

    private fun parseCandidates(o: JSONObject?): Candidates {
        if (o == null) return Candidates()
        return Candidates(
            snapshotBefore = parseSnapshot(o.optJSONObject("snapshot_before")),
            snapshotAfter = parseSnapshot(o.optJSONObject("snapshot_after")),
            relevantCells = parseIntList(o.optJSONArray("relevant_cells"))
        )
    }

    private fun parseSnapshot(o: JSONObject?): Map<String, Int> {
        if (o == null) return emptyMap()
        val out = HashMap<String, Int>()
        val it = o.keys()
        while (it.hasNext()) {
            val k = it.next()
            val v = o.opt(k)
            val n = when (v) {
                is Int -> v
                is Number -> v.toInt()
                is String -> v.toIntOrNull()
                else -> null
            }
            if (n != null) out[k] = n
        }
        return out
    }

    private fun parseApplications(arr: JSONArray?): List<ApplicationV2> {
        if (arr == null) return emptyList()
        val out = ArrayList<ApplicationV2>(arr.length())

        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue

            val id = optStringAny(o, "application_id", "applicationId").ifBlank { "" }
            val tid = optStringAny(o, "technique_id", "techniqueId").ifBlank { "" }
            if (id.isBlank() || tid.isBlank()) continue

            val name = optStringAny(o, "name_application", "nameApplication", "name").ifBlank { "" }
            val kind = optStringAny(o, "kind").ifBlank { "" }

            val supportObj = o.optJSONObject("support")
            val effectsObj = o.optJSONObject("effects")

            val supportV2 = supportObj?.let { sup ->
                SupportV2(
                    patternCells = parseIntList(sup.optJSONArray("pattern_cells")),
                    peerCells = parseIntList(sup.optJSONArray("peer_cells"))
                )
            }

            val effectsV2 = effectsObj?.let { eff ->
                // effects.eliminations expected to be a list of {cell_index/r/c/digit/...}
                val elim = parseEliminations(eff.optJSONArray("eliminations"))
                EffectsV2(eliminations = elim)
            }

            out += ApplicationV2(
                applicationId = id,
                techniqueId = tid,
                nameApplication = name,
                kind = kind,
                digits = parseIntList(o.optJSONArray("digits")),
                houses = parseHouses(o.optJSONArray("houses")),

                supportV2 = supportV2,
                effectsV2 = effectsV2,

                pattern = o.optJSONObject("pattern"),
                effects = effectsObj,
                support = supportObj,

                engineNative = o.optJSONObject("engine_native"),
                raw = o
            )
        }

        return out
    }

    private fun parseHintLadder(arr: JSONArray?): List<HintFrame> {
        if (arr == null) return emptyList()
        val out = ArrayList<HintFrame>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue
            val id = o.optString("frame_id")
            if (id.isBlank()) continue
            val text = o.optString("text")
            out += HintFrame(frameId = id, text = text, raw = o)
        }
        return out
    }

    private fun parseOverlayFrames(arr: JSONArray?): List<OverlayFrameV2> {
        if (arr == null) return emptyList()
        val out = ArrayList<OverlayFrameV2>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue
            val id = o.optString("frame_id")
            if (id.isBlank()) continue
            val style = o.optString("style").ifBlank { "full" }
            val hi = parseHighlights(o.optJSONArray("highlights"))
            out += OverlayFrameV2(frameId = id, style = style, highlights = hi, raw = o)
        }
        return out
    }

    private fun parseHighlights(arr: JSONArray?): List<OverlayHighlight> {
        if (arr == null) return emptyList()
        val out = ArrayList<OverlayHighlight>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue

            val kind = o.optString("kind")
            if (kind.isBlank()) continue

            val role = o.optString("role").ifBlank { null }
            val digit = o.optInt("digit", -1).takeIf { it in 1..9 }

            val ci = optIntAny(o, "cell_index", "cellIndex", default = -1)
            val cellIndex = ci.takeIf { it in 0..80 }

            val houseObj = o.optJSONObject("house")
            val house = if (houseObj != null) {
                val t = houseObj.optString("type")
                val idx = houseObj.optInt("index1to9", -1)
                if (t.isBlank() || idx !in 1..9) null else HouseRefV2(t, idx)
            } else null

            val fromIdx = optIntAny(o, "from_cell_index", "fromCellIndex", default = -1).takeIf { it in 0..80 }
            val toIdx = optIntAny(o, "to_cell_index", "toCellIndex", default = -1).takeIf { it in 0..80 }

            out += OverlayHighlight(
                kind = kind,
                cellIndex = cellIndex,
                role = role,
                digit = digit,
                house = house,
                fromCellIndex = fromIdx,
                toCellIndex = toIdx,
                raw = o
            )
        }
        return out
    }

    private fun parseTeaching(o: JSONObject?): Teaching {
        if (o == null) return Teaching()
        return Teaching(
            recognition = parseStringList(o.optJSONArray("recognition")),
            application = parseStringList(o.optJSONArray("application")),
            pitfalls = parseStringList(o.optJSONArray("pitfalls")),
            glossaryKeys = parseStringList(o.optJSONArray("glossary_keys"))
        )
    }

    private fun parseIntList(arr: JSONArray?): List<Int> {
        if (arr == null) return emptyList()
        val out = ArrayList<Int>(arr.length())
        for (i in 0 until arr.length()) {
            val v = arr.opt(i)
            val n = when (v) {
                is Int -> v
                is Number -> v.toInt()
                is String -> v.toIntOrNull()
                else -> null
            }
            if (n != null) out += n
        }
        return out
    }

    private fun parseStringList(arr: JSONArray?): List<String> {
        if (arr == null) return emptyList()
        val out = ArrayList<String>(arr.length())
        for (i in 0 until arr.length()) {
            val v = arr.optString(i)
            if (v.isNotBlank()) out += v.trim()
        }
        return out
    }

    private fun optIntAny(o: JSONObject, vararg keys: String, default: Int = -1): Int {
        for (k in keys) {
            if (o.has(k)) return o.optInt(k, default)
        }
        return default
    }

    private fun optStringAny(o: JSONObject, vararg keys: String): String {
        for (k in keys) {
            val v = o.optString(k).trim()
            if (v.isNotBlank()) return v
        }
        return ""
    }
}