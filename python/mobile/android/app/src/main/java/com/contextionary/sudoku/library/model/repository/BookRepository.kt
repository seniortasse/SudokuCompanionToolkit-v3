package com.contextionary.sudoku.library.repository

import com.contextionary.sudoku.library.model.CatalogBookSummary
import com.contextionary.sudoku.library.model.PuzzleCatalogRecord

class BookRepository(
    private val catalogRepository: LibraryCatalogRepository,
    private val puzzleRepository: PuzzleRepository,
) {
    fun getAllBooks(): List<CatalogBookSummary> {
        return catalogRepository.getBookSummaries().sortedBy { it.bookId }
    }

    fun getBooksForAisle(aisleId: String): List<CatalogBookSummary> {
        return getAllBooks().filter { it.aisleId == aisleId }
    }

    fun getBookById(bookId: String): CatalogBookSummary? {
        return getAllBooks().firstOrNull { it.bookId == bookId }
    }

    fun getPuzzlesForBook(bookId: String): List<PuzzleCatalogRecord> {
        return puzzleRepository.getAllPuzzles()
            .filter { it.bookId == bookId }
            .sortedWith(compareBy<PuzzleCatalogRecord> { it.sectionCode ?: "" }.thenBy { it.positionInSection ?: Int.MAX_VALUE })
    }
}