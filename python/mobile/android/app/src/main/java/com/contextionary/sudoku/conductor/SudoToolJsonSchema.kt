package com.contextionary.sudoku.conductor

import org.json.JSONArray
import org.json.JSONObject
import java.security.MessageDigest

/**
 * OpenAI Structured Outputs (json_schema strict) for Sudo tool planning.
 *
 * Output MUST be:
 * {
 *   "tool_calls": [ { "name": "...", "args": {...} }, ... ]
 * }
 *
 * IMPORTANT (strict schema pitfalls):
 * - In strict validator, if an object defines `properties`,
 *   then `required` must include EVERY key in `properties`.
 * - Therefore: do NOT model optional fields as extra properties.
 *   If a field is conceptually optional, represent it as nullable and still require it.
 */
object SudoToolJsonSchema {

    fun responseFormat(): JSONObject {
        return JSONObject()
            .put("type", "json_schema")
            .put(
                "json_schema",
                JSONObject()
                    .put("name", "sudo_policy")
                    .put("strict", true)
                    .put("schema", schemaObject())
            )
    }

    /** Telemetry/debug fingerprint for schema version used in a run. */
    fun schemaSha256(): String {
        val s = schemaObject().toString()
        val md = MessageDigest.getInstance("SHA-256")
        val digest = md.digest(s.toByteArray(Charsets.UTF_8))
        val hex = StringBuilder(digest.size * 2)
        for (b in digest) hex.append(String.format("%02x", b))
        return hex.toString()
    }

    private fun schemaObject(): JSONObject {
        return JSONObject()
            .put("type", "object")
            .put("additionalProperties", false)
            .put(
                "properties",
                JSONObject()
                    .put(
                        "tool_calls",
                        JSONObject()
                            .put("type", "array")
                            .put("minItems", 1)
                            .put("items", toolCallAnyOf())
                    )
            )
            .put("required", JSONArray().put("tool_calls"))
    }

    /** Use anyOf (supported) instead of oneOf. */
    private fun toolCallAnyOf(): JSONObject {
        return JSONObject().put(
            "anyOf",
            JSONArray()
                .put(replySchema())
                .put(askConfirmCellSchema())        // legacy
                .put(askConfirmCellRCSchema())      // ✅ Patch 1 (ask RC)
                .put(confirmCellValueSchema())      // ✅ Patch 1 (confirm index)
                .put(confirmCellValueRCSchema())    // ✅ Patch 1 (confirm RC)
                .put(proposeEditSchema())
                .put(applyUserEditSchema())
                .put(applyUserEditRCSchema())
                .put(recommendRetakeSchema())
                .put(recommendValidateSchema())
                .put(confirmInterpretationSchema())
                .put(askClarifyingQuestionSchema())
                .put(switchToTapSchema())
                .put(noopSchema())
        )
    }

    private fun replySchema(): JSONObject {
        val argsProps = JSONObject()
            .put("text", JSONObject().put("type", "string").put("minLength", 1))

        return JSONObject()
            .put("type", "object")
            .put("additionalProperties", false)
            .put(
                "properties",
                JSONObject()
                    .put("name", JSONObject().put("type", "string").put("enum", JSONArray().put(ToolCall.WireNames.REPLY)))
                    .put(
                        "args",
                        JSONObject()
                            .put("type", "object")
                            .put("additionalProperties", false)
                            .put("properties", argsProps)
                            .put("required", JSONArray().put("text"))
                    )
            )
            .put("required", JSONArray().put("name").put("args"))
    }

    private fun askConfirmCellSchema(): JSONObject {
        val argsProps = JSONObject()
            .put("cell_index", JSONObject().put("type", "integer").put("minimum", 0).put("maximum", 80))
            .put("prompt", JSONObject().put("type", "string").put("minLength", 1))

        return JSONObject()
            .put("type", "object")
            .put("additionalProperties", false)
            .put(
                "properties",
                JSONObject()
                    .put(
                        "name",
                        JSONObject().put("type", "string")
                            .put("enum", JSONArray().put(ToolCall.WireNames.ASK_CONFIRM_CELL))
                    )
                    .put(
                        "args",
                        JSONObject()
                            .put("type", "object")
                            .put("additionalProperties", false)
                            .put("properties", argsProps)
                            .put("required", JSONArray().put("cell_index").put("prompt"))
                    )
            )
            .put("required", JSONArray().put("name").put("args"))
    }

    /**
     * ✅ Patch 1:
     * ask_confirm_cell_rc(row,col,prompt) — preferred in GRID_SESSION.
     */
    private fun askConfirmCellRCSchema(): JSONObject {
        val argsProps = JSONObject()
            .put("row", JSONObject().put("type", "integer").put("minimum", 1).put("maximum", 9))
            .put("col", JSONObject().put("type", "integer").put("minimum", 1).put("maximum", 9))
            .put("prompt", JSONObject().put("type", "string").put("minLength", 1))

        return JSONObject()
            .put("type", "object")
            .put("additionalProperties", false)
            .put(
                "properties",
                JSONObject()
                    .put(
                        "name",
                        JSONObject().put("type", "string")
                            .put("enum", JSONArray().put(ToolCall.WireNames.ASK_CONFIRM_CELL_RC))
                    )
                    .put(
                        "args",
                        JSONObject()
                            .put("type", "object")
                            .put("additionalProperties", false)
                            .put("properties", argsProps)
                            .put("required", JSONArray().put("row").put("col").put("prompt"))
                    )
            )
            .put("required", JSONArray().put("name").put("args"))
    }

    /**
     * ✅ Patch 1:
     * confirm_cell_value(cell_index, digit)
     *
     * Used when the user has CONFIRMED what a cell contains.
     * digit allows 0..9 (0 = blank).
     */
    private fun confirmCellValueSchema(): JSONObject {
        val argsProps = JSONObject()
            .put("cell_index", JSONObject().put("type", "integer").put("minimum", 0).put("maximum", 80))
            .put("digit", JSONObject().put("type", "integer").put("minimum", 0).put("maximum", 9))

        return JSONObject()
            .put("type", "object")
            .put("additionalProperties", false)
            .put(
                "properties",
                JSONObject()
                    .put(
                        "name",
                        JSONObject().put("type", "string")
                            .put("enum", JSONArray().put(ToolCall.WireNames.CONFIRM_CELL_VALUE))
                    )
                    .put(
                        "args",
                        JSONObject()
                            .put("type", "object")
                            .put("additionalProperties", false)
                            .put("properties", argsProps)
                            .put("required", JSONArray().put("cell_index").put("digit"))
                    )
            )
            .put("required", JSONArray().put("name").put("args"))
    }

    /**
     * ✅ Patch 1:
     * confirm_cell_value_rc(row, col, digit)
     *
     * Preferred confirmation form in GRID_SESSION.
     * digit allows 0..9 (0 = blank).
     */
    private fun confirmCellValueRCSchema(): JSONObject {
        val argsProps = JSONObject()
            .put("row", JSONObject().put("type", "integer").put("minimum", 1).put("maximum", 9))
            .put("col", JSONObject().put("type", "integer").put("minimum", 1).put("maximum", 9))
            .put("digit", JSONObject().put("type", "integer").put("minimum", 0).put("maximum", 9))

        return JSONObject()
            .put("type", "object")
            .put("additionalProperties", false)
            .put(
                "properties",
                JSONObject()
                    .put(
                        "name",
                        JSONObject().put("type", "string")
                            .put("enum", JSONArray().put(ToolCall.WireNames.CONFIRM_CELL_VALUE_RC))
                    )
                    .put(
                        "args",
                        JSONObject()
                            .put("type", "object")
                            .put("additionalProperties", false)
                            .put("properties", argsProps)
                            .put("required", JSONArray().put("row").put("col").put("digit"))
                    )
            )
            .put("required", JSONArray().put("name").put("args"))
    }

    private fun proposeEditSchema(): JSONObject {
        val argsProps = JSONObject()
            .put("cell_index", JSONObject().put("type", "integer").put("minimum", 0).put("maximum", 80))
            .put("digit", JSONObject().put("type", "integer").put("minimum", 1).put("maximum", 9)) // keep 1..9
            .put("reason", JSONObject().put("type", "string").put("minLength", 1))
            .put("confidence", JSONObject().put("type", "number").put("minimum", 0).put("maximum", 1))

        return JSONObject()
            .put("type", "object")
            .put("additionalProperties", false)
            .put(
                "properties",
                JSONObject()
                    .put("name", JSONObject().put("type", "string").put("enum", JSONArray().put(ToolCall.WireNames.PROPOSE_EDIT)))
                    .put(
                        "args",
                        JSONObject()
                            .put("type", "object")
                            .put("additionalProperties", false)
                            .put("properties", argsProps)
                            .put("required", JSONArray().put("cell_index").put("digit").put("reason").put("confidence"))
                    )
            )
            .put("required", JSONArray().put("name").put("args"))
    }

    private fun applyUserEditSchema(): JSONObject {
        val argsProps = JSONObject()
            .put("cell_index", JSONObject().put("type", "integer").put("minimum", 0).put("maximum", 80))
            // ✅ allow blank
            .put("digit", JSONObject().put("type", "integer").put("minimum", 0).put("maximum", 9))
            .put("source", JSONObject().put("type", "string").put("minLength", 1))

        return JSONObject()
            .put("type", "object")
            .put("additionalProperties", false)
            .put(
                "properties",
                JSONObject()
                    .put("name", JSONObject().put("type", "string").put("enum", JSONArray().put(ToolCall.WireNames.APPLY_USER_EDIT)))
                    .put(
                        "args",
                        JSONObject()
                            .put("type", "object")
                            .put("additionalProperties", false)
                            .put("properties", argsProps)
                            .put("required", JSONArray().put("cell_index").put("digit").put("source"))
                    )
            )
            .put("required", JSONArray().put("name").put("args"))
    }

    private fun applyUserEditRCSchema(): JSONObject {
        val argsProps = JSONObject()
            .put("row", JSONObject().put("type", "integer").put("minimum", 1).put("maximum", 9))
            .put("col", JSONObject().put("type", "integer").put("minimum", 1).put("maximum", 9))
            // ✅ allow blank
            .put("digit", JSONObject().put("type", "integer").put("minimum", 0).put("maximum", 9))
            .put("source", JSONObject().put("type", "string").put("minLength", 1))

        return JSONObject()
            .put("type", "object")
            .put("additionalProperties", false)
            .put(
                "properties",
                JSONObject()
                    .put("name", JSONObject().put("type", "string").put("enum", JSONArray().put(ToolCall.WireNames.APPLY_USER_EDIT_RC)))
                    .put(
                        "args",
                        JSONObject()
                            .put("type", "object")
                            .put("additionalProperties", false)
                            .put("properties", argsProps)
                            .put("required", JSONArray().put("row").put("col").put("digit").put("source"))
                    )
            )
            .put("required", JSONArray().put("name").put("args"))
    }

    private fun recommendRetakeSchema(): JSONObject {
        val argsProps = JSONObject()
            .put("strength", JSONObject().put("type", "string").put("minLength", 1))
            .put("reason", JSONObject().put("type", "string").put("minLength", 1))

        return JSONObject()
            .put("type", "object")
            .put("additionalProperties", false)
            .put(
                "properties",
                JSONObject()
                    .put("name", JSONObject().put("type", "string").put("enum", JSONArray().put(ToolCall.WireNames.RECOMMEND_RETAKE)))
                    .put(
                        "args",
                        JSONObject()
                            .put("type", "object")
                            .put("additionalProperties", false)
                            .put("properties", argsProps)
                            .put("required", JSONArray().put("strength").put("reason"))
                    )
            )
            .put("required", JSONArray().put("name").put("args"))
    }

    private fun recommendValidateSchema(): JSONObject {
        val argsProps = JSONObject()
        return JSONObject()
            .put("type", "object")
            .put("additionalProperties", false)
            .put(
                "properties",
                JSONObject()
                    .put("name", JSONObject().put("type", "string").put("enum", JSONArray().put(ToolCall.WireNames.RECOMMEND_VALIDATE)))
                    .put(
                        "args",
                        JSONObject()
                            .put("type", "object")
                            .put("additionalProperties", false)
                            .put("properties", argsProps)
                            .put("required", JSONArray())
                    )
            )
            .put("required", JSONArray().put("name").put("args"))
    }

    private fun confirmInterpretationSchema(): JSONObject {
        // In strict mode: if present in properties, must be required.
        // So we model optional fields as nullable but required.
        val intOrNull = JSONObject().put("type", JSONArray().put("integer").put("null"))
        val numOrNull = JSONObject().put("type", JSONArray().put("number").put("null"))

        val argsProps = JSONObject()
            .put("row", intOrNull)
            .put("col", intOrNull)
            .put("digit", intOrNull) // still 1..9 or null by runtime rules
            .put("prompt", JSONObject().put("type", "string").put("minLength", 1))
            .put("confidence", numOrNull)

        return JSONObject()
            .put("type", "object")
            .put("additionalProperties", false)
            .put(
                "properties",
                JSONObject()
                    .put("name", JSONObject().put("type", "string").put("enum", JSONArray().put(ToolCall.WireNames.CONFIRM_INTERPRETATION)))
                    .put(
                        "args",
                        JSONObject()
                            .put("type", "object")
                            .put("additionalProperties", false)
                            .put("properties", argsProps)
                            .put("required", JSONArray().put("row").put("col").put("digit").put("prompt").put("confidence"))
                    )
            )
            .put("required", JSONArray().put("name").put("args"))
    }

    private fun askClarifyingQuestionSchema(): JSONObject {
        val argsProps = JSONObject()
            .put("kind", JSONObject().put("type", "string").put("minLength", 1))
            .put("prompt", JSONObject().put("type", "string").put("minLength", 1))

        return JSONObject()
            .put("type", "object")
            .put("additionalProperties", false)
            .put(
                "properties",
                JSONObject()
                    .put("name", JSONObject().put("type", "string").put("enum", JSONArray().put(ToolCall.WireNames.ASK_CLARIFYING_QUESTION)))
                    .put(
                        "args",
                        JSONObject()
                            .put("type", "object")
                            .put("additionalProperties", false)
                            .put("properties", argsProps)
                            .put("required", JSONArray().put("kind").put("prompt"))
                    )
            )
            .put("required", JSONArray().put("name").put("args"))
    }

    private fun switchToTapSchema(): JSONObject {
        val argsProps = JSONObject()
            .put("prompt", JSONObject().put("type", "string").put("minLength", 1))

        return JSONObject()
            .put("type", "object")
            .put("additionalProperties", false)
            .put(
                "properties",
                JSONObject()
                    .put("name", JSONObject().put("type", "string").put("enum", JSONArray().put(ToolCall.WireNames.SWITCH_TO_TAP)))
                    .put(
                        "args",
                        JSONObject()
                            .put("type", "object")
                            .put("additionalProperties", false)
                            .put("properties", argsProps)
                            .put("required", JSONArray().put("prompt"))
                    )
            )
            .put("required", JSONArray().put("name").put("args"))
    }

    private fun noopSchema(): JSONObject {
        val argsProps = JSONObject()
        return JSONObject()
            .put("type", "object")
            .put("additionalProperties", false)
            .put(
                "properties",
                JSONObject()
                    .put("name", JSONObject().put("type", "string").put("enum", JSONArray().put(ToolCall.WireNames.NOOP)))
                    .put(
                        "args",
                        JSONObject()
                            .put("type", "object")
                            .put("additionalProperties", false)
                            .put("properties", argsProps)
                            .put("required", JSONArray())
                    )
            )
            .put("required", JSONArray().put("name").put("args"))
    }
}