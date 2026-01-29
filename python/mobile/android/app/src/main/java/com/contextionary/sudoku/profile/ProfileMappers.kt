package com.contextionary.sudoku.profile

import com.contextionary.sudoku.profile.PlayerProfileSnapshot

/**
 * Map our app-local [UserProfile] to the profile-facing [PlayerProfileSnapshot].
 */
fun UserProfile.toSnapshot(): PlayerProfileSnapshot =
    PlayerProfileSnapshot(
        name = this.name,
        locale = this.preferredLanguage,
        favoriteDifficulty = this.favoriteDifficulty,
        interests = this.interestsList()
    )