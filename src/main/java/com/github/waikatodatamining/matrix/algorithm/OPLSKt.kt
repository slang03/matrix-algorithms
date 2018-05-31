package com.github.waikatodatamining.matrix.algorithm

import Jama.Matrix
import com.github.waikatodatamining.matrix.core.*
import com.github.waikatodatamining.matrix.core.MatrixHelper.l2VectorNorm

open class OPLSKt : AbstractSingleResponsePLS() {
    override fun canPredict() = true

    private val serialVersionUID = -6097279189841762321L

    /** the P matrix  */
    protected lateinit var Porth: Matrix

    /** the T matrix  */
    protected lateinit var Torth: Matrix

    /** the W matrix  */
    protected lateinit var Worth: Matrix

    /** Data with orthogonal signal components removed  */
    protected lateinit var Xosc: Matrix

    /** Base PLS that is trained on the cleaned data  */
    lateinit var basePLS: AbstractPLS


    override fun initialize() {
        super.initialize()
        basePLS = PLS1()

    }

    /**
     * Returns the all the available matrices.
     *
     * @return the names of the matrices
     */
    override fun getMatrixNames(): Array<String> {
        return arrayOf("P_orth", "W_orth", "T_orth")
    }

    /**
     * Returns the matrix with the specified name.
     *
     * @param name the name of the matrix
     * @return the matrix, null if not available
     */
    override fun getMatrix(name: String): Matrix? {
        when (name) {
            "P_orth" -> return Porth
            "W_orth" -> return Worth
            "T_orth" -> return Torth
            else -> return null
        }
    }

    /**
     * Whether the algorithm supports return of loadings.
     *
     * @return true if supported
     * @see .getLoadings
     */
    override fun hasLoadings(): Boolean {
        return true
    }

    /**
     * Returns the loadings, if available.
     *
     * @return the loadings, null if not available
     */
    override fun getLoadings(): Matrix? {
        return getMatrix("P_orth")
    }

    /**
     * Get the inverse of the squared l2 norm.
     * @param v Input vector
     * @return 1.0 / norm2(v)^2
     */
    protected fun invL2Squared(v: Matrix): Double {
        val l2 = l2VectorNorm(v)
        return 1.0 / (l2 * l2)
    }

    override fun doTransform(predictors: Matrix): Matrix {
        val T = predictors * Worth
        val Xorth = T * Porth.T
        return predictors - Xorth
    }


    /**
     * Performs predictions on the data.
     *
     * @param predictors the input data
     * @return the transformed data and the predictions
     * @throws Exception if analysis fails
     */
    override fun doPerformPredictions(predictors: Matrix): Matrix {
        val Xtransformed = transform(predictors)
        return basePLS.predict(Xtransformed)
    }

    /**
     * Initializes using the provided data.
     *
     * @param predictors the input data
     * @param response   the dependent variable(s)
     * @return null if successful, otherwise error message
     */
    override fun doPerformInitialization(predictors: Matrix, response: Matrix): String? {
        var w: Matrix
        var wOrth: Matrix
        var t: Matrix
        var tOrth: Matrix
        var p: Matrix
        var pOrth: Matrix

        var X = predictors.copy()
        var Xtrans = X.T
        val y = response

        // init
        Worth = Matrix(predictors.columnDimension, numComponents)
        Porth = Matrix(predictors.columnDimension, numComponents)
        Torth = Matrix(predictors.rowDimension, numComponents)

        w = Xtrans * y / y.l2Sq()
        w = w.normalized()

        for (j in 0 until numComponents) {
            // Calculate scores vector
            t = X * w / w.l2Sq()

            // Calculate loadings of X
            p = Xtrans * t / t.l2Sq()

            // Orthogonalize weight
            wOrth = p - w * (w.T * p / w.l2Sq())
            wOrth = wOrth.normalized()
            tOrth = X * wOrth / wOrth.l2Sq()
            pOrth = Xtrans * tOrth / tOrth.l2Sq()

            // Remove orthogonal components from X
            X -= tOrth * pOrth.T
            Xtrans = X.T

            // Store results
            Worth[j] = wOrth
            Torth[j] = tOrth
            Porth[j] = pOrth
        }

        Xosc = X.copy()
        basePLS.initialize(this.doTransform(predictors), response)

        return null
    }
}