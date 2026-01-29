package com.contextionary.sudoku.conversation

/**
 * Persistent store contract for turns.
 *
 * IMPORTANT: This must be declared ONLY here (to avoid redeclaration errors).
 */
interface TurnStore {

    /**
     * Insert or update a TurnRecord by (sessionId, turnId).
     */
    fun upsert(turn: TurnRecord)

    /**
     * Return the latest turn for a session (highest turnId), or null.
     *
     * Your TurnLifecycleManager and RecoveryController call this as loadLast(...).
     */
    fun loadLast(sessionId: SessionId): TurnRecord?

    /**
     * Returns up to [limit] most recent turns (chronological: oldest -> newest).
     *
     * PromptBuilder relies on this ordering to serialize history deterministically.
     */
    fun loadRecent(sessionId: SessionId, limit: Int): List<TurnRecord>

    /**
     * Remove all turns for a session.
     */
    fun clearSession(sessionId: SessionId)
}