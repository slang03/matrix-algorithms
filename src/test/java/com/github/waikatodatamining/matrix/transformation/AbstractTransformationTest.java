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
 * AbstractTransformationTest.java
 * Copyright (C) 2018 University of Waikato, Hamilton, NZ
 */

package com.github.waikatodatamining.matrix.transformation;

import Jama.Matrix;
import com.github.waikatodatamining.matrix.core.MatrixHelper;
import com.github.waikatodatamining.matrix.test.AbstractTestCase;
import com.github.waikatodatamining.matrix.test.TestHelper;
import com.github.waikatodatamining.matrix.test.TmpFile;

/**
 * Ancestor for transformer tests.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
public abstract class AbstractTransformationTest
  extends AbstractTestCase {

  /**
   * Constructs the test case. Called by subclasses.
   *
   * @param name the name of the test
   */
  public AbstractTransformationTest(String name) {
    super(name);
  }

  /**
   * Returns the test helper class to use.
   *
   * @return		the helper class instance
   */
  @Override
  protected TestHelper newTestHelper() {
    return new TestHelper(this, "com/github/waikatodatamining/matrix/transformation/data");
  }

  /**
   * Returns the ignored line indices to use in the regression test.
   *
   * @return		the setups
   */
  protected int[] getRegressionIgnoredLineIndices() {
    return new int[0];
  }

  /**
   * Processes the input data and returns the processed data.
   *
   * @param data	the data to work on
   * @param scheme	the scheme to process the data with
   * @return		the processed data
   */
  protected Matrix process(Matrix data, AbstractTransformation scheme) {
    return scheme.transform(data);
  }

  /**
   * Check if inverse transform of transformed data equals the original data.
   * @param data Input data
   * @param scheme The transformation scheme
   * @return True if data == invTrans(trans(data))
   */
  protected boolean checkInvertTransform(Matrix data, AbstractTransformation scheme){
    Matrix transformed = scheme.transform(data);
    Matrix restored = scheme.inverseTransform(transformed);
    return MatrixHelper.equal(data, restored, 1e-8);
  }

  /**
   * Returns the filenames (without path) of the input data files to use
   * in the regression test.
   *
   * @return		the filenames
   */
  protected abstract String[] getRegressionInputFiles();

  /**
   * Returns the setups to use in the regression test.
   *
   * @return		the setups
   */
  protected abstract AbstractTransformation[] getRegressionSetups();

  /**
   * Loads the CSV file.
   *
   * @param filename	the CSV file
   * @return		the matrix
   */
  protected Matrix load(String filename) {
    try {
      m_TestHelper.copyResourceToTmp(filename);
      return MatrixHelper.read(new TmpFile(filename).getAbsolutePath(), true, ',');
    }
    catch (Exception e) {
      fail("Failed to read: " + filename + "\n" + e);
      return null;
    }
  }

  /**
   * Saves the matrix to the specified file.
   *
   * @param data	the data to save
   * @param filename	the file to save to
   * @return		true if successful
   */
  protected boolean save(Matrix data, String filename) {
    try {
      MatrixHelper.write(data, m_TestHelper.getTmpLocationFromResource(filename), true, '\t', 6);
      return true;
    }
    catch (Exception e) {
      return false;
    }
  }

  /**
   * Creates an output filename based on the input filename.
   *
   * @param input	the input filename (no path)
   * @param no		the number of the test
   * @return		the generated output filename (no path)
   */
  protected String createOutputFilename(String input, int no) {
    String	result;
    int		index;
    String	ext;

    ext = "-out" + no;

    index = input.lastIndexOf('.');
    if (index == -1) {
      result = input + ext;
    }
    else {
      result  = input.substring(0, index);
      result += ext;
      result += input.substring(index);
    }

    return result;
  }

  /**
   * Compares the processed data against previously saved output data.
   */
  public void testRegression() {
    Matrix 			data;
    Matrix			processed;
    boolean			ok;
    String			regression;
    int				i;
    String[]			input;
    AbstractTransformation[]	setups;
    String[]			output;
    TmpFile[]			outputFiles;
    int[]			ignored;

    if (m_NoRegressionTest)
      return;

    input   = getRegressionInputFiles();
    output  = new String[input.length];
    setups  = getRegressionSetups();
    ignored = getRegressionIgnoredLineIndices();
    assertEquals("Number of files and setups differ!", input.length, setups.length);

    // process data
    for (i = 0; i < input.length; i++) {
      data = load(input[i]);
      assertNotNull("Could not load data for regression test from " + input[i], data);

      assertTrue("Inverse transformation did not restore original data", checkInvertTransform(data, setups[i]));

      processed = process(data, setups[i]);
      assertNotNull("Failed to process data?", processed);

      output[i] = createOutputFilename(input[i], i);
      ok        = save(processed, output[i]);
      assertTrue("Failed to save regression data?", ok);
    }

    // test regression
    outputFiles = new TmpFile[output.length];
    for (i = 0; i < output.length; i++)
      outputFiles[i] = new TmpFile(output[i]);
    regression = m_Regression.compare(outputFiles, ignored);
    assertNull("Output differs:\n" + regression, regression);

    // remove output, clean up scheme
    for (i = 0; i < output.length; i++) {
      m_TestHelper.deleteFileFromTmp(output[i]);
    }
  }
}
