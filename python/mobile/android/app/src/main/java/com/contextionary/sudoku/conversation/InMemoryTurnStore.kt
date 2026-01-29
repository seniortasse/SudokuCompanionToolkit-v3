package com.contextionary.sudoku.conversation

import java.util.concurrent.ConcurrentHashMap

class InMemoryTurnStore : TurnStore {

    private val bySession = ConcurrentHashMap<String, MutableList<TurnRecord>>()

    override fun upsert(record: TurnRecord) {
        val list = bySession.getOrPut(record.sessionId) { mutableListOf() }
        synchronized(list) {
            val idx = list.indexOfFirst { it.turnId == record.turnId }
            if (idx >= 0) list[idx] = record else list.add(record)
            list.sortBy { it.turnId }
        }
    }

    override fun loadLast(sessionId: String): TurnRecord? {
        val list = bySession[sessionId] ?: return null
        synchronized(list) { return list.maxByOrNull { it.turnId } }
    }

    override fun loadRecent(sessionId: String, limit: Int): List<TurnRecord> {
        val list = bySession[sessionId] ?: return emptyList()
        synchronized(list) {
            return list.sortedByDescending { it.turnId }.take(limit).sortedBy { it.turnId }
        }
    }

    override fun clearSession(sessionId: String) {
        bySession.remove(sessionId)
    }
}