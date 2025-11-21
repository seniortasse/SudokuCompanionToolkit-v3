package com.contextionary.sudoku

import android.os.Bundle
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

class ResultActivity : AppCompatActivity() {

    // Keep mutable copies so we can edit in-place
    private lateinit var digits: IntArray
    private lateinit var confs: FloatArray
    private var changedIndices: List<Int> = emptyList()
    private var unresolvedIndices: MutableList<Int> = mutableListOf()

    override fun onCreate(savedInstanceState: Bundle?) {
        // Use the black theme for a seamless fade-in (declared in manifest)
        setTheme(R.style.Theme_SudokuCompanion_Results)
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)

        val view = findViewById<SudokuResultView>(R.id.sudokuView)
        val btnRetake = findViewById<MaterialButton>(R.id.btnRetake)
        val btnKeep = findViewById<MaterialButton>(R.id.btnKeep)

        // ---- 1) Read data from Intent ----

        // Digits: required; if missing, default to empty grid.
        digits = (intent.getIntArrayExtra("digits") ?: IntArray(81)).copyOf()

        // Confidences: optional; if missing, default to 1.0f everywhere.
        confs = intent.getFloatArrayExtra("confidences")
            ?: FloatArray(81) { 1.0f }

        // Changed / unresolved indices: optional; if not provided, stay empty.
        val changedFromIntent = intent.getIntArrayExtra("changedIndices")
        changedIndices = changedFromIntent?.toList() ?: emptyList()

        val unresolvedFromIntent = intent.getIntArrayExtra("unresolvedIndices")
        unresolvedIndices = unresolvedFromIntent?.toMutableList() ?: mutableListOf()

        // ---- 2) Initialize the view ----
        view.setDigitsAndConfidences(digits, confs)
        view.setLogicAnnotations(
            changed = changedIndices,
            unresolved = unresolvedIndices
        )

        // ---- 3) Wire up cell tap handling ----
        view.setOnCellClickListener(object : SudokuResultView.OnCellClickListener {
            override fun onCellClicked(row: Int, col: Int) {
                showDigitPickerDialog(row, col, view)
            }
        })

        // ---- 4) Buttons ----
        btnRetake.setOnClickListener {
            setResult(RESULT_CANCELED)
            finish()
            // avoid animations on exit
            overridePendingTransition(0, 0)
        }

        btnKeep.setOnClickListener {
            setResult(RESULT_OK)
            finish()
            // avoid animations on exit
            overridePendingTransition(0, 0)
        }
    }

    /**
     * Show a simple dialog to let the user choose:
     * - Clear (set digit to 0)
     * - Digits 1..9
     */
    private fun showDigitPickerDialog(
        row: Int,
        col: Int,
        view: SudokuResultView
    ) {
        val idx = row * 9 + col

        // Label options; index 0 = Clear, 1..9 = digits
        val labels = arrayOf(
            "Clear",
            "1", "2", "3", "4", "5", "6", "7", "8", "9"
        )

        AlertDialog.Builder(this)
            .setTitle("Edit cell r${row + 1}c${col + 1}")
            .setItems(labels) { dialog, which ->
                when (which) {
                    0 -> { // Clear
                        digits[idx] = 0
                        // You may choose to keep confidence as-is; or set to 1.0f as "user-confirmed"
                        confs[idx] = 1.0f
                    }
                    in 1..9 -> {
                        val newDigit = which // 1..9
                        digits[idx] = newDigit
                        // Mark as confidently set by user
                        confs[idx] = 1.0f
                    }
                    else -> {
                        // Should not happen
                    }
                }

                // Remove this cell from unresolved list, since the user has now explicitly edited it.
                unresolvedIndices.remove(idx)

                // Re-apply to the view
                view.setDigitsAndConfidences(digits, confs)
                view.setLogicAnnotations(
                    changed = changedIndices,
                    unresolved = unresolvedIndices
                )

                dialog.dismiss()
            }
            .setNegativeButton("Cancel") { dialog, _ ->
                dialog.dismiss()
            }
            .show()
    }
}