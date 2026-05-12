package com.contextionary.sudoku.conductor.policy

import java.time.Instant

private inline fun <reified T : Enum<T>> mergerEnumValueOrNull(raw: String?): T? {
    val s = raw?.trim()?.takeIf { it.isNotEmpty() } ?: return null
    return enumValues<T>().firstOrNull { it.name.equals(s, ignoreCase = true) }
}

object RelationshipMemoryMergerV1 {

    fun merge(
        base: RelationshipMemoryV1,
        delta: RelationshipDeltaV1
    ): RelationshipMemoryV1 {
        if (delta.observations.isEmpty() && delta.candidateUpdates.isEmpty()) return base

        var out = base
        for (u in delta.candidateUpdates) {
            out = applyCandidateUpdate(out, u)
        }
        out = applyObservations(out, delta.observations)
        out = refreshIntegrity(out, delta)
        return out
    }

    private fun applyCandidateUpdate(
        base: RelationshipMemoryV1,
        u: RelationshipCandidateUpdateV1
    ): RelationshipMemoryV1 {
        return when (u.bucket.trim()) {
            "relationship_tone_bond" -> base.copy(
                relationshipToneBond = applyRelationshipToneBond(base.relationshipToneBond, u)
            )

            "communication_speech_style" -> base.copy(
                communicationSpeechStyle = applyCommunicationSpeechStyle(base.communicationSpeechStyle, u)
            )

            "learning_explanation_model" -> base.copy(
                learningExplanationModel = applyLearningExplanationModel(base.learningExplanationModel, u)
            )

            "sudoku_knowledge_technique_map" -> base.copy(
                sudokuKnowledgeTechniqueMap = applySudokuKnowledgeTechniqueMap(base.sudokuKnowledgeTechniqueMap, u)
            )

            "solving_mindset_cognitive_style" -> base.copy(
                solvingMindsetCognitiveStyle = applySolvingMindsetCognitiveStyle(base.solvingMindsetCognitiveStyle, u)
            )

            "world_context_solving_reality" -> base.copy(
                worldContextSolvingReality = applyWorldContextSolvingReality(base.worldContextSolvingReality, u)
            )

            "personal_language_meaning_hooks" -> base.copy(
                personalLanguageMeaningHooks = applyPersonalLanguageMeaningHooks(base.personalLanguageMeaningHooks, u)
            )

            "interaction_history_memory_integrity" -> base.copy(
                interactionHistoryMemoryIntegrity = applyInteractionHistoryMemoryIntegrity(
                    base.interactionHistoryMemoryIntegrity,
                    u
                )
            )

            else -> base
        }
    }

    private fun applyRelationshipToneBond(
        base: RelationshipToneBondV1,
        u: RelationshipCandidateUpdateV1
    ): RelationshipToneBondV1 {
        return when (u.key.trim()) {
            "relationship_tone" -> base.copy(
                relationshipTone = mergerEnumValueOrNull<RelationshipToneV1>(u.value) ?: base.relationshipTone
            )

            "familiarity_preference" -> base.copy(
                familiarityPreference = mergerEnumValueOrNull<FamiliarityPreferenceV1>(u.value) ?: base.familiarityPreference
            )

            "name_usage_preference" -> base.copy(nameUsagePreference = u.value)
            "encouragement_style" -> base.copy(encouragementStyle = u.value)

            "humor_preference" -> base.copy(
                humorPreference = mergerEnumValueOrNull<HumorPreferenceV1>(u.value) ?: base.humorPreference
            )

            "trust_builder", "trust_builders" -> base.copy(
                trustBuilders = addUnique(base.trustBuilders, u.value)
            )

            "warmth_preference", "warmth_preferences" -> base.copy(
                warmthPreferences = addUnique(base.warmthPreferences, u.value)
            )

            "bond_note", "bond_notes" -> base.copy(
                bondNotes = addUnique(base.bondNotes, u.value)
            )

            else -> base
        }
    }

    private fun applyCommunicationSpeechStyle(
        base: CommunicationSpeechStyleV1,
        u: RelationshipCandidateUpdateV1
    ): CommunicationSpeechStyleV1 {
        return when (u.key.trim()) {
            "pace_preference" -> base.copy(
                pacePreference = mergerEnumValueOrNull<PacePreferenceV1>(u.value) ?: base.pacePreference
            )

            "verbosity_preference" -> base.copy(verbosityPreference = u.value)
            "speech_rhythm_preference" -> base.copy(speechRhythmPreference = u.value)
            "question_tolerance" -> base.copy(questionTolerance = u.value)
            "confirmation_style_preference" -> base.copy(confirmationStylePreference = u.value)
            "repetition_sensitivity" -> base.copy(repetitionSensitivity = u.value)
            "jargon_tolerance" -> base.copy(jargonTolerance = u.value)
            "cta_style_preference" -> base.copy(ctaStylePreference = u.value)

            "avoid_speech_pattern", "avoid_speech_patterns" -> base.copy(
                avoidSpeechPatterns = addUnique(base.avoidSpeechPatterns, u.value)
            )

            else -> base
        }
    }

    private fun applyLearningExplanationModel(
        base: LearningExplanationModelV1,
        u: RelationshipCandidateUpdateV1
    ): LearningExplanationModelV1 {
        return when (u.key.trim()) {
            "learning_preference", "learning_preferences" -> {
                val e = mergerEnumValueOrNull<ExplanationStyleV1>(u.value)
                if (e != null) base.copy(learningPreferences = addUniqueEnum(base.learningPreferences, e)) else base
            }

            "proof_preference" -> base.copy(
                proofPreference = mergerEnumValueOrNull<ProofPreferenceV1>(u.value) ?: base.proofPreference
            )

            "setup_preference" -> base.copy(setupPreference = u.value)
            "confrontation_preference" -> base.copy(confrontationPreference = u.value)
            "resolution_preference" -> base.copy(resolutionPreference = u.value)
            "analogy_preference" -> base.copy(analogyPreference = u.value)

            "clarity_trigger", "clarity_triggers" -> base.copy(
                clarityTriggers = addUnique(base.clarityTriggers, u.value)
            )

            "confusion_trigger", "confusion_triggers" -> base.copy(
                confusionTriggers = addUnique(base.confusionTriggers, u.value)
            )

            "metaphor_domain" -> base.copy(
                metaphorPolicy = base.metaphorPolicy.copy(
                    allowed = true,
                    domains = addUnique(base.metaphorPolicy.domains, u.value)
                )
            )

            "metaphor_frequency" -> base.copy(
                metaphorPolicy = base.metaphorPolicy.copy(
                    allowed = true,
                    frequency = u.value
                )
            )

            "metaphor_allowed" -> base.copy(
                metaphorPolicy = base.metaphorPolicy.copy(
                    allowed = u.value.equals("true", ignoreCase = true) ||
                            u.value.equals("yes", ignoreCase = true) ||
                            u.value.equals("allowed", ignoreCase = true)
                )
            )

            "metaphor_natural_only" -> base.copy(
                metaphorPolicy = base.metaphorPolicy.copy(
                    naturalOnly = !u.value.equals("false", ignoreCase = true)
                )
            )

            else -> base
        }
    }

    private fun applySudokuKnowledgeTechniqueMap(
        base: SudokuKnowledgeTechniqueMapV1,
        u: RelationshipCandidateUpdateV1
    ): SudokuKnowledgeTechniqueMapV1 {
        return when (u.key.trim()) {
            "sudoku_identity" -> base.copy(sudokuIdentity = u.value)
            "skill_self_view" -> base.copy(skillSelfView = u.value)

            "recently_learned_technique", "recently_learned_techniques" -> base.copy(
                recentlyLearnedTechniques = addUnique(base.recentlyLearnedTechniques, u.value)
            )

            "fragile_technique", "fragile_techniques" -> base.copy(
                fragileTechniques = addUnique(base.fragileTechniques, u.value)
            )

            "challenging_technique", "challenging_techniques" -> base.copy(
                challengingTechniques = addUnique(base.challengingTechniques, u.value)
            )

            "favorite_technique", "favorite_techniques" -> base.copy(
                favoriteTechniques = addUnique(base.favoriteTechniques, u.value)
            )

            "mastered_pattern", "mastered_patterns" -> base.copy(
                masteredPatterns = addUnique(base.masteredPatterns, u.value)
            )

            "teaching_priority_gap", "teaching_priority_gaps" -> base.copy(
                teachingPriorityGaps = addUnique(base.teachingPriorityGaps, u.value)
            )

            else -> {
                if (u.key.startsWith("technique_familiarity:", ignoreCase = true)) {
                    val technique = u.key.substringAfter(":", "").trim()
                    if (technique.isEmpty()) return base
                    val fam = mergerEnumValueOrNull<TechniqueFamiliarityV1>(u.value) ?: return base
                    val updated = upsertTechniqueComfort(base.techniqueComfort, technique, fam, null)
                    base.copy(techniqueComfort = updated)
                } else {
                    base
                }
            }
        }
    }

    private fun applySolvingMindsetCognitiveStyle(
        base: SolvingMindsetCognitiveStyleV1,
        u: RelationshipCandidateUpdateV1
    ): SolvingMindsetCognitiveStyleV1 {
        return when (u.key.trim()) {
            "thinking_style" -> base.copy(thinkingStyle = addUnique(base.thinkingStyle, u.value))
            "decision_style" -> base.copy(decisionStyle = u.value)
            "attention_style" -> base.copy(attentionStyle = u.value)

            "mental_model_note", "mental_model_notes" -> base.copy(
                mentalModelNotes = addUnique(base.mentalModelNotes, u.value)
            )

            "common_reasoning_move", "common_reasoning_moves" -> base.copy(
                commonReasoningMoves = addUnique(base.commonReasoningMoves, u.value)
            )

            "common_blind_spot", "common_blind_spots" -> base.copy(
                commonBlindSpots = addUnique(base.commonBlindSpots, u.value)
            )

            "error_tendency", "error_tendencies" -> base.copy(
                errorTendencies = addUnique(base.errorTendencies, u.value)
            )

            "confidence_pattern" -> base.copy(confidencePattern = u.value)
            "autonomy_preference" -> base.copy(autonomyPreference = u.value)
            else -> base
        }
    }

    private fun applyWorldContextSolvingReality(
        base: WorldContextSolvingRealityV1,
        u: RelationshipCandidateUpdateV1
    ): WorldContextSolvingRealityV1 {
        return when (u.key.trim()) {
            "solving_medium" -> base.copy(solvingMedium = u.value)

            "usual_environment", "usual_environments" -> base.copy(
                usualEnvironments = addUnique(base.usualEnvironments, u.value)
            )

            "interaction_constraint", "interaction_constraints" -> base.copy(
                interactionConstraints = addUnique(base.interactionConstraints, u.value)
            )

            "session_style" -> base.copy(sessionStyle = u.value)

            "physical_reality_note", "physical_reality_notes" -> base.copy(
                physicalRealityNotes = addUnique(base.physicalRealityNotes, u.value)
            )

            "validation_need", "validation_needs" -> base.copy(
                validationNeeds = addUnique(base.validationNeeds, u.value)
            )

            "routine_pattern", "routine_patterns" -> base.copy(
                routinePatterns = addUnique(base.routinePatterns, u.value)
            )

            else -> base
        }
    }

    private fun applyPersonalLanguageMeaningHooks(
        base: PersonalLanguageMeaningHooksV1,
        u: RelationshipCandidateUpdateV1
    ): PersonalLanguageMeaningHooksV1 {
        return when (u.key.trim()) {
            "user_jargon" -> base.copy(userJargon = addUnique(base.userJargon, u.value))
            "preferred_term", "preferred_terms" -> base.copy(
                preferredTerms = addUnique(base.preferredTerms, u.value)
            )

            "disliked_term", "disliked_terms" -> base.copy(
                dislikedTerms = addUnique(base.dislikedTerms, u.value)
            )

            "mental_label", "mental_labels" -> base.copy(
                mentalLabels = addUnique(base.mentalLabels, u.value)
            )

            "metaphor_domain", "metaphor_domains" -> base.copy(
                metaphorDomains = addUnique(base.metaphorDomains, u.value)
            )

            "meaning_hook", "meaning_hooks" -> base.copy(
                meaningHooks = addUnique(base.meaningHooks, u.value)
            )

            "language_register" -> base.copy(languageRegister = u.value)
            "bilingual_or_language_notes" -> base.copy(bilingualOrLanguageNotes = u.value)
            else -> base
        }
    }

    private fun applyInteractionHistoryMemoryIntegrity(
        base: InteractionHistoryMemoryIntegrityV1,
        u: RelationshipCandidateUpdateV1
    ): InteractionHistoryMemoryIntegrityV1 {
        return when (u.key.trim()) {
            "user_experience_summary" -> base.copy(userExperienceSummary = u.value)

            "current_priority", "current_priorities" -> base.copy(
                currentPriorities = addUnique(base.currentPriorities, u.value)
            )

            "recent_growth_edge", "recent_growth_edges" -> base.copy(
                recentGrowthEdges = addUnique(base.recentGrowthEdges, u.value)
            )

            "recent_friction_edge", "recent_friction_edges" -> base.copy(
                recentFrictionEdges = addUnique(base.recentFrictionEdges, u.value)
            )

            "staleness_flag", "staleness_flags" -> base.copy(
                stalenessFlags = addUnique(base.stalenessFlags, u.value)
            )

            "source_note", "source_notes" -> base.copy(
                sourceNotes = addUnique(base.sourceNotes, u.value)
            )

            else -> base
        }
    }

    private fun applyObservations(
        base: RelationshipMemoryV1,
        observations: List<RelationshipObservationV1>
    ): RelationshipMemoryV1 {
        if (observations.isEmpty()) return base
        val notes = observations.map { it.note.trim() }.filter { it.isNotEmpty() }
        if (notes.isEmpty()) return base

        val integrity = base.interactionHistoryMemoryIntegrity.copy(
            sourceNotes = addAllUnique(base.interactionHistoryMemoryIntegrity.sourceNotes, notes)
        )
        return base.copy(interactionHistoryMemoryIntegrity = integrity)
    }

    private fun refreshIntegrity(
        base: RelationshipMemoryV1,
        delta: RelationshipDeltaV1
    ): RelationshipMemoryV1 {
        val existing = base.interactionHistoryMemoryIntegrity

        val confidenceEntries = mutableListOf<BucketConfidenceV1>().apply {
            addAll(existing.confidenceByBucket)
            val touchedBuckets = delta.candidateUpdates.map { it.bucket.trim() }.filter { it.isNotEmpty() }.distinct()
            for (bucket in touchedBuckets) {
                upsertBucketConfidence(this, bucket, delta.confidence)
            }
        }

        val evidenceEntries = mutableListOf<EvidenceCountEntryV1>().apply {
            addAll(existing.evidenceCounts)
            bumpEvidence(this, "relationship_observations", delta.observations.size)
            for (u in delta.candidateUpdates) {
                bumpEvidence(this, "${u.bucket}:${u.key}", 1)
            }
        }

        val summary = buildUserExperienceSummary(base)

        return base.copy(
            interactionHistoryMemoryIntegrity = existing.copy(
                userExperienceSummary = summary,
                confidenceByBucket = confidenceEntries,
                evidenceCounts = evidenceEntries,
                lastRefreshedAt = Instant.now().toString()
            )
        )
    }

    private fun buildUserExperienceSummary(memory: RelationshipMemoryV1): String {
        val tone = memory.relationshipToneBond.relationshipTone?.name
            ?.lowercase()
            ?.replace('_', ' ')
        val pace = memory.communicationSpeechStyle.pacePreference?.name
            ?.lowercase()
            ?.replace('_', ' ')
        val proof = memory.learningExplanationModel.proofPreference?.name
            ?.lowercase()
            ?.replace('_', ' ')

        val parts = mutableListOf<String>()
        if (!tone.isNullOrBlank()) parts += "Best companion tone appears to be $tone."
        if (!pace.isNullOrBlank()) parts += "Preferred pacing appears to be $pace."
        if (!proof.isNullOrBlank()) parts += "Proof detail preference appears to be $proof."
        if (memory.personalLanguageMeaningHooks.userJargon.isNotEmpty()) {
            parts += "User-specific wording is emerging."
        }
        if (memory.sudokuKnowledgeTechniqueMap.techniqueComfort.isNotEmpty()) {
            parts += "Technique comfort map is beginning to form."
        }
        return parts.joinToString(" ").ifBlank {
            "Relationship memory is still sparse; prefer conservative adaptation."
        }
    }

    private fun upsertTechniqueComfort(
        base: List<TechniqueComfortEntryV1>,
        technique: String,
        familiarity: TechniqueFamiliarityV1,
        notes: String?
    ): List<TechniqueComfortEntryV1> {
        val idx = base.indexOfFirst { it.technique.equals(technique, ignoreCase = true) }
        return if (idx >= 0) {
            base.toMutableList().apply {
                val old = this[idx]
                this[idx] = old.copy(
                    familiarity = familiarity,
                    notes = notes ?: old.notes
                )
            }
        } else {
            base + TechniqueComfortEntryV1(
                technique = technique,
                familiarity = familiarity,
                notes = notes
            )
        }
    }

    private fun upsertBucketConfidence(
        xs: MutableList<BucketConfidenceV1>,
        bucket: String,
        confidence: ConfidenceLevelV1
    ) {
        val idx = xs.indexOfFirst { it.bucket.equals(bucket, ignoreCase = true) }
        if (idx >= 0) {
            val old = xs[idx]
            xs[idx] = old.copy(confidence = maxConfidence(old.confidence, confidence))
        } else {
            xs += BucketConfidenceV1(bucket = bucket, confidence = confidence)
        }
    }

    private fun bumpEvidence(
        xs: MutableList<EvidenceCountEntryV1>,
        key: String,
        inc: Int
    ) {
        if (inc <= 0) return
        val idx = xs.indexOfFirst { it.key.equals(key, ignoreCase = true) }
        if (idx >= 0) {
            val old = xs[idx]
            xs[idx] = old.copy(count = old.count + inc)
        } else {
            xs += EvidenceCountEntryV1(key = key, count = inc)
        }
    }

    private fun maxConfidence(a: ConfidenceLevelV1, b: ConfidenceLevelV1): ConfidenceLevelV1 {
        val rank = mapOf(
            ConfidenceLevelV1.LOW to 1,
            ConfidenceLevelV1.MEDIUM to 2,
            ConfidenceLevelV1.HIGH to 3
        )
        return if ((rank[b] ?: 0) > (rank[a] ?: 0)) b else a
    }

    private fun addUnique(xs: List<String>, x: String): List<String> {
        val v = x.trim()
        if (v.isEmpty()) return xs
        return if (xs.any { it.equals(v, ignoreCase = true) }) xs else xs + v
    }

    private fun addAllUnique(xs: List<String>, ys: List<String>): List<String> {
        var out = xs
        ys.forEach { out = addUnique(out, it) }
        return out
    }

    private fun <T> addUniqueEnum(xs: List<T>, x: T): List<T> {
        return if (xs.contains(x)) xs else xs + x
    }
}