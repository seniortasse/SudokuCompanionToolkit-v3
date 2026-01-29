package com.contextionary.sudoku.logic

/**
 * COMPAT ONLY.
 * MainActivity still references these while you migrate fully to tool-calls-only.
 * Safe to keep temporarily; the network path does NOT use LLMRawAction anymore.
 */

data class LLMRawAction(
    val type: String,
    val cell: String? = null,
    val digit: Int? = null,
    val options: List<Int>? = null
)

data class LLMRawResponse(
    val assistant_message: String,
    val action: LLMRawAction
)

sealed class LLMAction {
    data class ChangeCell(val row: Int, val col: Int, val digit: Int) : LLMAction()
    data class AskUserConfirmation(val row: Int, val col: Int, val options: List<Int>) : LLMAction()
    object RetakePhoto : LLMAction()
    object NoAction : LLMAction()
    object ValidateGrid : LLMAction()
}