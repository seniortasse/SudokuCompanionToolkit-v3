package com.contextionary.sudoku.logic

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject

/**
 * Real LLM client using OpenAI's chat completions endpoint.
 *
 * It:
 *  - sends system + grid summary to the model
 *  - asks for JSON output (assistant_message + action)
 *  - logs HTTP errors and response bodies for debugging
 */
class RealSudokuLLMClient(
    private val apiKey: String,
    private val model: String
) : SudokuLLMClient {

    private val httpClient = OkHttpClient()
    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()

    override suspend fun sendGridUpdate(
        systemPrompt: String,
        developerPrompt: String,
        userMessage: String
    ): LLMRawResponse = withContext(Dispatchers.IO) {
        try {
            // For now we treat developerPrompt as a big “context blob”.
            // We send it plus the latest userMessage as a single user message.
            val mergedUserContent = buildString {
                appendLine("Here is the latest grid state description:")
                appendLine()
                appendLine(developerPrompt)
                if (userMessage.isNotBlank()) {
                    appendLine()
                    appendLine("User just said:")
                    appendLine(userMessage)
                }
                appendLine()
                appendLine(
                    "Respond ONLY with a single JSON object of the form " +
                            "{\"assistant_message\": \"...\", \"action\": { ... }}."
                )
            }

            val messages = JSONArray().apply {
                put(
                    JSONObject().apply {
                        put("role", "system")
                        put("content", systemPrompt)
                    }
                )
                put(
                    JSONObject().apply {
                        put("role", "user")
                        put("content", mergedUserContent)
                    }
                )
            }

            val payload = JSONObject().apply {
                put("model", model)
                put(
                    "response_format",
                    JSONObject().apply { put("type", "json_object") }
                )
                put("messages", messages)
            }

            val requestBody = payload.toString().toRequestBody(jsonMediaType)

            val request = Request.Builder()
                .url("https://api.openai.com/v1/chat/completions")
                .addHeader("Authorization", "Bearer $apiKey")
                .addHeader("Content-Type", "application/json")
                .post(requestBody)
                .build()

            val response = httpClient.newCall(request).execute()
            val bodyString = response.body?.string() ?: ""

            // ---- HTTP-level logging ----
            if (!response.isSuccessful) {
                Log.e(
                    "SudokuLLM",
                    "OpenAI error: HTTP ${response.code} - $bodyString"
                )
                return@withContext LLMRawResponse(
                    assistant_message = "I had trouble reaching the server while looking at your puzzle. Please try again in a moment.",
                    action = LLMRawAction(type = "no_action")
                )
            }

            // Log the raw success response for now so we can inspect it.
            Log.i("SudokuLLM", "OpenAI raw response: $bodyString")

            // ---- Parse the assistant message (which should itself be JSON) ----
            val root = JSONObject(bodyString)
            val choices = root.getJSONArray("choices")
            if (choices.length() == 0) {
                Log.e("SudokuLLM", "OpenAI response has no choices")
                return@withContext LLMRawResponse(
                    assistant_message = "I had trouble understanding the server’s reply. Please try again.",
                    action = LLMRawAction(type = "no_action")
                )
            }

            val content = choices
                .getJSONObject(0)
                .getJSONObject("message")
                .getString("content")
                .trim()

            Log.i("SudokuLLM", "OpenAI message content: $content")

            // content should be a JSON object string with assistant_message + action
            val contentJson = JSONObject(content)

            val assistantMessage = contentJson.optString(
                "assistant_message",
                "I'm here and ready to help with your puzzle."
            )

            val actionJson = contentJson.optJSONObject("action")
            val rawAction = if (actionJson != null) {
                val type = actionJson.optString("type", "no_action")
                val cell = if (actionJson.has("cell")) {
                    actionJson.optString("cell", null)
                } else {
                    null
                }

                val digit: Int? =
                    if (actionJson.has("digit")) actionJson.optInt("digit")
                    else null

                val options: List<Int>? =
                    if (actionJson.has("options")) {
                        val arr = actionJson.getJSONArray("options")
                        List(arr.length()) { idx -> arr.getInt(idx) }
                    } else {
                        null
                    }

                LLMRawAction(
                    type = type,
                    cell = cell,
                    digit = digit,
                    options = options
                )
            } else {
                LLMRawAction(type = "no_action")
            }

            return@withContext LLMRawResponse(
                assistant_message = assistantMessage,
                action = rawAction
            )
        } catch (e: Exception) {
            Log.e("SudokuLLM", "Error calling OpenAI", e)
            return@withContext LLMRawResponse(
                assistant_message = "I had trouble reaching the server while looking at your puzzle. Please try again.",
                action = LLMRawAction(type = "no_action")
            )
        }
    }
}