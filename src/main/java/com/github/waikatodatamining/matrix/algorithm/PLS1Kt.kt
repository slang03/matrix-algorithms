/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * PLS1.java
 * Copyright (C) 2018 University of Waikato, Hamilton, NZ
 */

package com.github.waikatodatamining.matrix.algorithm

import Jama.Matrix
import com.github.waikatodatamining.matrix.core.*
import com.github.waikatodatamining.matrix.core.MatrixEntryType.*

/**
 * PLS1 algorithm.
 * <br></br>
 * See here:
 * [Statmaster Module 7](https://web.archive.org/web/20081001154431/http://statmaster.sdu.dk:80/courses/ST02/module07/module.pdf)
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
class PLS1Kt : AbstractSingleResponsePLS() {

    /** the regression vector "r-hat"  */
    protected var rHat: Matrix? = null

    /** the P matrix  */
    protected var P: Matrix? = null

    /** the W matrix  */
    protected var W: Matrix? = null

    /** the b-hat vector  */
    protected var bHat: Matrix? = null

    /**
     * Resets the member variables.
     */
    override fun reset() {
        super.reset()

        rHat = null
        P = null
        W = null
        bHat = null
    }

    /**
     * Returns the all the available matrices.
     *
     * @return        the names of the matrices
     */
    override fun getMatrixNames(): Array<String> {
        return arrayOf("r_hat", "P", "W", "b_hat")
    }

    /**
     * Returns the matrix with the specified name.
     *
     * @param name    the name of the matrix
     * @return        the matrix, null if not available
     */
    override fun getMatrix(name: String): Matrix? {
        when (name) {
            "RegVector" -> return rHat
            "P" -> return P
            "W" -> return W
            "b_hat" -> return bHat
            else -> return null
        }
    }

    /**
     * Whether the algorithm supports return of loadings.
     *
     * @return        true if supported
     * @see .getLoadings
     */
    override fun hasLoadings(): Boolean {
        return true
    }

    /**
     * Returns the loadings, if available.
     *
     * @return        the loadings, null if not available
     */
    override fun getLoadings(): Matrix? {
        return getMatrix("P")
    }

    /**
     * Initializes using the provided data.
     *
     * @param predictors the input data
     * @param response   the dependent variable(s)
     * @throws Exception if analysis fails
     * @return null if successful, otherwise error message
     */
    @Throws(Exception::class)
    override fun doPerformInitialization(predictors: Matrix, response: Matrix): String? {
        var predictors = predictors
        var response = response
        val Xtrans: Matrix
        val W: Matrix
        var w: Matrix
        val T: Matrix
        var t: Matrix
        var tTrans: Matrix
        val P: Matrix
        var p: Matrix
        var pTrans: Matrix
        var b: Double
        val bHat: Matrix
        val tmp: Matrix

        Xtrans = predictors.T

        // init
        W = Matrix(predictors.columnDimension, numComponents)
        P = Matrix(predictors.columnDimension, numComponents)
        T = Matrix(predictors.rowDimension, numComponents)
        bHat = Matrix(numComponents, 1)

        for (j in 0 until numComponents) {
            // 1. step: wj
            w = Xtrans * response
            w = w.normalized()
            W[j] = w

            // 2. step: tj
            t = predictors * w
            tTrans = t.T
            T[j] = t

            // 3. step: ^bj
            b = (tTrans * response).get(0, 0) / t.l2Sq()
            bHat.set(j, 0, b)

            // 4. step: pj
            p = Xtrans * t / t.l2Sq()
            pTrans = p.T
            P[j] = p
            // 5. step: Xj+1
            predictors -= t * pTrans
            response -= t * b
        }

        // W*(P^T*W)^-1
        tmp = W * (P.T * W).inverse()

        // factor = W*(P^T*W)^-1 * bHat
        rHat = tmp * bHat

        // save matrices
        this.P = P
        this.W = W
        this.bHat = bHat

        return null
    }

    /**
     * Transforms the data.
     *
     * @param predictors the input data
     * @throws Exception if analysis fails
     * @return the transformed data and the predictions
     */
    @Throws(Exception::class)
    override fun doTransform(predictors: Matrix): Matrix {
        val result: Matrix
        var T: Matrix
        var t: Matrix
        var x: Matrix
        var X: Matrix
        var i: Int
        var j: Int

        result = Matrix(predictors.rowDimension, numComponents)

        i = 0
        while (i < predictors.rowDimension) {
            // work on each row
            x = predictors[ROW, i]
            X = Matrix(1, numComponents)
            T = Matrix(1, numComponents)

            j = 0
            while (j < numComponents) {
                X[j] = x
                // 1. step: tj = xj * wj
                t = x * W!![COL, j]
                T[j] = t
                // 2. step: xj+1 = xj - tj*pj^T (tj is 1x1 matrix!)
                x = x.minus(P!![COL, j].T * t)
                j++
            }
            result[ROW, i] = T
            i++
        }

        return result
    }

    /**
     * Returns whether the algorithm can make predictions.
     *
     * @return        true if can make predictions
     */
    override fun canPredict(): Boolean {
        return true
    }

    /**
     * Performs predictions on the data.
     *
     * @param predictors the input data
     * @throws Exception if analysis fails
     * @return the transformed data and the predictions
     */
    @Throws(Exception::class)
    override fun doPerformPredictions(predictors: Matrix): Matrix {
        val result: Matrix
        var T: Matrix
        var t: Matrix
        var x: Matrix
        var X: Matrix
        var i: Int
        var j: Int

        result = Matrix(predictors.rowDimension, 1)

        i = 0
        while (i < predictors.rowDimension) {
            // work on each row
            x = predictors[ROW, i]
            X = Matrix(1, numComponents)
            T = Matrix(1, numComponents)

            j = 0
            while (j < numComponents) {
                X[j] = x
                // 1. step: tj = xj * wj
                t = x * W!![COL, j]
                T[j] = t
                // 2. step: xj+1 = xj - tj*pj^T (tj is 1x1 matrix!)
                x -= P!![COL, j].T * t.get(0, 0)
                j++
            }

            result[i, 0] = (T * bHat!!).get(0, 0)
            i++
        }

        return result
    }

    companion object {

        private val serialVersionUID = 4899661745515419256L
    }
}