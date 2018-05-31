package com.github.waikatodatamining.matrix.core

import Jama.Matrix

enum class MatrixEntryType {
    ROW,
    COL
}

operator fun Matrix.set(column: Int, vector: Matrix) {
    MatrixHelper.setColumnVector(vector, this, column)
}

operator fun Matrix.set(type: MatrixEntryType, idx: Int, vector: Matrix) {
    when (type) {
        MatrixEntryType.ROW -> MatrixHelper.setRowVector(vector, this, idx)
        MatrixEntryType.COL -> MatrixHelper.setColumnVector(vector, this, idx)
    }
}

fun Matrix.normalized(): Matrix {
    val copy = this.copy()
    MatrixHelper.normalizeVector(copy)
    return copy
}


fun Matrix.l2() = MatrixHelper.l2VectorNorm(this)

fun Matrix.l2Sq(): Double {
    val l2 = this.l2()
    return l2 * l2
}


val Matrix.T: Matrix
    get() = this.transpose()

