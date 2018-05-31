package com.github.waikatodatamining.matrix.core

import Jama.Matrix

enum class MatrixEntryType {
    ROW,
    COL
}

/**
 * Implement bracket accessor: matrix[j] = other
 */
operator fun Matrix.set(column: Int, vector: Matrix) {
    MatrixHelper.setColumnVector(vector, this, column)
}


/**
 * Implement bracket accessor: matrix[i,j] = number
 */
operator fun Matrix.set(type: MatrixEntryType, idx: Int, vector: Matrix) {
    when (type) {
        MatrixEntryType.ROW -> MatrixHelper.setRowVector(vector, this, idx)
        MatrixEntryType.COL -> MatrixHelper.setColumnVector(vector, this, idx)
    }
}

/**
 * Implement bracket accessor: matrix[i,j] = number
 */
operator fun Matrix.get(type: MatrixEntryType, idx: Int) : Matrix = when (type) {
    MatrixEntryType.ROW -> MatrixHelper.rowAsVector(this, idx)
    MatrixEntryType.COL -> MatrixHelper.columnAsVector(this, idx)
}

/**
 * Add div opartor: matrix / double
 */
operator fun Matrix.div(value: Double): Matrix = this * (1.0 / value)

/**
 * Get this vector as a normalized vector
 */
fun Matrix.normalized(): Matrix {
    val copy = this.copy()
    MatrixHelper.normalizeVector(copy)
    return copy
}

/**
 * Get l2 norm
 */
fun Matrix.l2() = MatrixHelper.l2VectorNorm(this)

/**
 * Get l2-norm squared
 */
fun Matrix.l2Sq(): Double {
    val l2 = this.l2()
    return l2 * l2
}

/**
 * Add transposition as member
 */
val Matrix.T: Matrix
    get() = this.transpose()

