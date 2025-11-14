package com.contextionary.sudoku

object Logx {
    private const val PREFIX = "SC"
    fun d(tag: String, vararg kv: Pair<String, Any?>) {
        val body = kv.joinToString(" ") { "${it.first}=${it.second}" }
        android.util.Log.d("$PREFIX/$tag", body)
    }
    fun e(tag: String, err: Throwable, vararg kv: Pair<String, Any?>) {
        val body = kv.joinToString(" ") { "${it.first}=${it.second}" }
        android.util.Log.e("$PREFIX/$tag", "$body cause=${err.message}", err)
    }
}