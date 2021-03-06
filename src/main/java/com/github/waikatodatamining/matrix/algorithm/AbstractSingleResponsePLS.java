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
 * AbstractSingleResponsePLS.java
 * Copyright (C) 2018 University of Waikato, Hamilton, NZ
 */

package com.github.waikatodatamining.matrix.algorithm;

import Jama.Matrix;
import com.github.waikatodatamining.matrix.core.MatrixHelper;
import com.github.waikatodatamining.matrix.transformation.AbstractTransformation;
import com.github.waikatodatamining.matrix.transformation.Center;
import com.github.waikatodatamining.matrix.transformation.Standardize;

/**
 * Ancestor for PLS algorithms that work on a single response variable.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
public abstract class AbstractSingleResponsePLS
  extends AbstractPLS {

  private static final long serialVersionUID = -8160023117935320371L;

  /** the class mean. */
  protected double m_ClassMean;

  /** the class stddev. */
  protected double m_ClassStdDev;

  /** the transformation for the predictors. */
  protected AbstractTransformation m_TransPredictors;

  /** the transformation for the response. */
  protected AbstractTransformation m_TransResponse;

  /**
   * Resets the member variables.
   */
  @Override
  protected void reset() {
    super.reset();

    m_ClassMean       = Double.NaN;
    m_ClassStdDev     = Double.NaN;
    m_TransPredictors = null;
    m_TransResponse   = null;
  }

  /**
   * Hook method for checking the data before training.
   *
   * @param predictors	the input data
   * @param response 	the dependent variable(s)
   * @return		null if successful, otherwise error message
   */
  @Override
  protected String check(Matrix predictors, Matrix response) {
    String	result;

    result = super.check(predictors, response);

    if (result == null) {
      if (response.getColumnDimension() != 1)
	result = "Algorithm requires exactly one response variable, found: " + response.getColumnDimension();
    }

    return result;
  }

  /**
   * Initializes using the provided data.
   *
   * @param predictors	the input data
   * @param response 	the dependent variable(s)
   * @return		null if successful, otherwise error message
   * @throws Exception	if analysis fails
   */
  protected String doInitialize(Matrix predictors, Matrix response) throws Exception {
    String	result;

    switch (m_PreprocessingType) {
      case CENTER:
	m_ClassMean       = MatrixHelper.mean(response, 0);
	m_ClassStdDev     = 1;
	m_TransPredictors = new Center();
	m_TransResponse   = new Center();
	break;
      case STANDARDIZE:
	m_ClassMean       = MatrixHelper.mean(response, 0);
	m_ClassStdDev     = MatrixHelper.stdev(response, 0);
	m_TransPredictors = new Standardize();
	m_TransResponse   = new Standardize();
	break;
      case NONE:
	m_ClassMean       = 0;
	m_ClassStdDev     = 1;
	m_TransPredictors = null;
	m_TransResponse   = null;
	break;
      default:
	throw new IllegalStateException("Unhandled preprocessing type; " + m_PreprocessingType);
    }

    if (m_TransPredictors != null) {
      m_TransPredictors.configure(predictors);
      predictors = m_TransPredictors.transform(predictors);
    }
    if (m_TransResponse != null) {
      m_TransResponse.configure(response);
      response = m_TransResponse.transform(response);
    }

    result = doPerformInitialization(predictors, response);

    return result;
  }

  /**
   * Performs predictions on the data.
   *
   * @param predictors the input data
   * @throws Exception if analysis fails
   * @return the transformed data and the predictions
   */
  @Override
  protected Matrix doPredict(Matrix predictors) throws Exception {
    Matrix	result;
    int		i;

    result = doPerformPredictions(predictors);
    if (m_TransResponse != null) {
      for (i = 0; i < result.getRowDimension(); i++)
        result.set(i, 0, result.get(i, 0) * m_ClassStdDev + m_ClassMean);
    }

    return result;
  }
}
