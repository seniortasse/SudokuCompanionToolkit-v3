package com.contextionary.sudoku

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import com.contextionary.sudoku.SudokuResultView

class ResultActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        // Use the black theme for a seamless fade-in (declared in manifest)
        setTheme(R.style.Theme_SudokuCompanion_Results)
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)

        val view = findViewById<SudokuResultView>(R.id.sudokuView)
        val btnRetake = findViewById<MaterialButton>(R.id.btnRetake)
        val btnKeep = findViewById<MaterialButton>(R.id.btnKeep)

        // Read digits from Intent; 0s won't render in the view
        val digits: IntArray = intent.getIntArrayExtra("digits") ?: IntArray(81)
        view.setDigits(digits)

        btnRetake.setOnClickListener {
            setResult(RESULT_CANCELED)
            finish()

            //avoid animations on exit
            overridePendingTransition(0, 0)
        }

        btnKeep.setOnClickListener {
            setResult(RESULT_OK)
            finish()

            //avoid animations on exit
            overridePendingTransition(0, 0)
        }
    }
}