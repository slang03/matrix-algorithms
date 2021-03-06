package com.github.waikatodatamining.matrix.algorithm;

import Jama.Matrix;
import com.github.waikatodatamining.matrix.core.MatrixHelper;
import com.github.waikatodatamining.matrix.transformation.Center;
import com.github.waikatodatamining.matrix.transformation.kernel.AbstractKernel;
import com.github.waikatodatamining.matrix.transformation.kernel.RBFKernel;

import static com.github.waikatodatamining.matrix.core.MatrixHelper.l2VectorNorm;

/**
 * Kernel Partial Least Squares algorithm.
 * <br>
 * See here:
 * <a href="http://www.jmlr.org/papers/volume2/rosipal01a/rosipal01a.pdf">Kernel Partial Least Squares Regression in Reproducing
 * Kernel Hilbert Space</a>
 *
 * @author Steven Lang
 */
public class KernelPLS extends AbstractMultiResponsePLS {

  private static final long serialVersionUID = -2760078672082710402L;

  public static final int SEED = 0;

  /** Calibration data in feature space */
  protected Matrix m_K_orig;

  protected Matrix m_K_deflated;

  /** Scores on K */
  protected Matrix m_T;

  /** Scores on Y */
  protected Matrix m_U;

  /** Loadings on K */
  protected Matrix m_P;

  /** Loadings on Y */
  protected Matrix m_Q;

  /** Partial regression matrix */
  protected Matrix m_B_RHS;

  /** Training points */
  protected Matrix m_X;

  /** Kernel for feature transformation */
  protected AbstractKernel m_Kernel;


  /** Inner NIPALS loop improvement tolerance */
  protected double m_Tol;

  /** Inner NIPALS loop maximum number of iterations */
  protected int m_MaxIter;

  /** Center X transformation */
  protected Center m_CenterX;

  /** Center Y transformation */
  protected Center m_CenterY;

  @Override
  protected void initialize() {
    super.initialize();
    setKernel(new RBFKernel());
    setTol(1e-6);
    setMaxIter(500);
    m_CenterX = new Center();
    m_CenterY = new Center();
  }

  public AbstractKernel getKernel() {
    return m_Kernel;
  }

  public void setKernel(AbstractKernel kernel) {
    this.m_Kernel = kernel;
  }

  public int getMaxIter() {
    return m_MaxIter;
  }

  public void setMaxIter(int maxIter) {
    this.m_MaxIter = maxIter;
  }

  public double getTol() {
    return m_Tol;
  }

  public void setTol(double tol) {
    this.m_Tol = tol;
  }

  @Override
  protected int getMinColumnsResponse() {
    return 1;
  }

  @Override
  protected int getMaxColumnsResponse() {
    return -1;
  }

  @Override
  protected String doPerformInitialization(Matrix predictors, Matrix response) throws Exception {
    Matrix Y, I, t, u, q, w;


    // Init
    int numComponents = getNumComponents();
    m_X = predictors;
    m_X = m_CenterX.transform(m_X);
    Y = response;
    Y = m_CenterY.transform(Y);

    int numRows = m_X.getRowDimension();
    int numClasses = Y.getColumnDimension();

    q = new Matrix(numClasses, 1);
    t = new Matrix(numRows, 1);
    w = new Matrix(numRows, 1);
    I = Matrix.identity(numRows, numRows);

    m_T = new Matrix(numRows, numComponents);
    m_U = new Matrix(numRows, numComponents);
    m_P = new Matrix(numRows, numComponents);
    m_Q = new Matrix(numClasses, numComponents);
    m_K_orig = m_Kernel.applyMatrix(m_X, m_X);
    m_K_orig = centralizeInKernelSpace(m_K_orig);
    m_K_deflated = m_K_orig.copy();

    for (int currentComponent = 0; currentComponent < numComponents; currentComponent++) {
      int iterations = 0;
      Matrix uOld;
      u = MatrixHelper.randn(numRows, 1, SEED + currentComponent);
      double iterationChange = m_Tol * 10;

      // Repeat 1) - 3) until convergence: either change of u is lower than m_Tol or maximum
      // number of iterations has been reached (m_MaxIter)
      while (iterationChange > m_Tol && iterations < m_MaxIter) {
	// 1)
	t = m_K_deflated.times(u);
	w = t.copy();
	MatrixHelper.normalizeVector(t);

	// 2)
	q = Y.transpose().times(t);

	// 3)
	uOld = u;
	u = Y.times(q);
	MatrixHelper.normalizeVector(u);

	// Update stopping conditions
	iterations++;
	iterationChange = l2VectorNorm(u.minus(uOld));
      }

      // Deflate
      Matrix ttTrans = t.times(t.transpose());
      Matrix part = I.minus(ttTrans);
      m_K_deflated = part.times(m_K_deflated).times(part);
      Y = Y.minus(t.times(q.transpose()));
      Matrix p = m_K_deflated.transpose().times(w).times(1.0 / w.transpose().times(w).get(0, 0));


      // Store u,t,c,p
      MatrixHelper.setColumnVector(t, m_T, currentComponent);
      MatrixHelper.setColumnVector(u, m_U, currentComponent);
      MatrixHelper.setColumnVector(q, m_Q, currentComponent);
      MatrixHelper.setColumnVector(p, m_P, currentComponent);
    }

    // Calculate right hand side of the regression matrix B
    Matrix tTtimesKtimesU = m_T.transpose().times(m_K_orig).times(m_U);
    Matrix inv = tTtimesKtimesU.inverse();
    m_B_RHS = inv.times(m_Q.transpose());
    return null;
  }

  /**
   * Centralize a kernel matrix in the kernel space via:
   * K <- (I - 1/n * 1_n * 1_n^T) * K * (I - 1/n * 1_n * 1_n^T)
   *
   * @param K Kernel matrix
   * @return Centralised kernel matrix
   */
  protected Matrix centralizeInKernelSpace(Matrix K) {
    int n = m_X.getRowDimension();
    Matrix I = Matrix.identity(n, n);
    Matrix one = new Matrix(n, 1, 1.0);

    // Centralize in kernel space
    Matrix part = I.minus(one.times(one.transpose()).times(1.0 / n));
    return part.times(K).times(part);
  }

  @Override
  protected Matrix doPerformPredictions(Matrix predictors) {
    Matrix K_t = doTransform(predictors);
    Matrix Y_hat = K_t.times(m_B_RHS);
    Y_hat = m_CenterY.inverseTransform(Y_hat);
    return Y_hat;
  }

  @Override
  protected Matrix doTransform(Matrix predictors) {
    Matrix predictorsCentered = m_CenterX.transform(predictors);
    Matrix K_t = m_Kernel.applyMatrix(predictorsCentered, m_X);
    K_t = centralizeInKernelSpace(K_t);

    return K_t.times(m_U);
  }

  @Override
  public String[] getMatrixNames() {
    return new String[]{"K", "T", "U", "P", "Q"};
  }

  @Override
  public Matrix getMatrix(String name) {
    switch (name) {
      case "K":
	return m_K_deflated;
      case "T":
	return m_T;
      case "U":
	return m_U;
      case "P":
	return m_P;
      case "Q":
	return m_Q;
    }
    return null;
  }

  @Override
  public boolean hasLoadings() {
    return true;
  }

  @Override
  protected void reset() {
    super.reset();
    m_K_orig = null;
    m_K_deflated = null;
    m_T = null;
    m_U = null;
    m_P = null;
    m_Q = null;
    m_B_RHS = null;
    m_X = null;
    m_CenterX = new Center();
    m_CenterY = new Center();
  }

  @Override
  public Matrix getLoadings() {
    return m_T;
  }

  @Override
  public boolean canPredict() {
    return true;
  }


}
