package com.github.waikatodatamining.matrix.algorithm

import Jama.Matrix
import com.github.waikatodatamining.matrix.core.*
import com.github.waikatodatamining.matrix.core.MatrixEntryType.COL
import com.github.waikatodatamining.matrix.core.MatrixHelper.l2VectorNorm

open class OPLSKt : AbstractSingleResponsePLS() {
    override fun canPredict() = true

    private val serialVersionUID = -6097279189841762321L

    /** the P matrix  */
    protected lateinit var m_Porth: Matrix

    /** the T matrix  */
    protected lateinit var m_Torth: Matrix

    /** the W matrix  */
    protected lateinit var m_Worth: Matrix

    /** Data with orthogonal signal components removed  */
    protected lateinit var m_Xosc: Matrix

    /** Base PLS that is trained on the cleaned data  */
    protected lateinit var m_BasePLS: AbstractPLS


    override fun initialize() {
        super.initialize()
        m_BasePLS = PLS1()

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
            "P_orth" -> return m_Porth
            "W_orth" -> return m_Worth
            "T_orth" -> return m_Torth
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
        val T = predictors * m_Worth
        val Xorth = T * m_Porth.T
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
        return m_BasePLS.predict(Xtransformed)
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
        m_Worth = Matrix(predictors.columnDimension, numComponents)
        m_Porth = Matrix(predictors.columnDimension, numComponents)
        m_Torth = Matrix(predictors.rowDimension, numComponents)

        w = Xtrans * y * (1.0 / y.l2Sq())
        w = w.normalized()

        for (currentComponent in 0 until numComponents) {

            // Calculate scores vector
            t = X * w * (1.0 / w.l2Sq())

            // Calculate loadings of X
            p = Xtrans * t * (1.0 / t.l2Sq())

            // Orthogonalize weight
            wOrth = p - w * (w.T * p * (1.0 / w.l2Sq()))[0, 0]
            wOrth = wOrth.normalized()
            tOrth = X * wOrth * (1.0 / wOrth.l2Sq())
            pOrth = Xtrans * tOrth * (1.0 / tOrth.l2Sq())

            // Remove orthogonal components from X
            X -= tOrth * pOrth.T
            Xtrans = X.T

            // Store results
            m_Worth[COL, currentComponent] = wOrth
            m_Torth[COL, currentComponent] = tOrth
            m_Porth[COL, currentComponent] = pOrth
        }

        m_Xosc = X.copy()
        m_BasePLS.initialize(this.doTransform(predictors), response)

        return null
    }
}