using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
//using Microsoft.VisualStudio.TestTools.UnitTesting;
using TestNeuralNetwork;
using System.Reflection;
using System.IO;
using CommonLib;
using NUnit.Framework;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Data.Text;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworkTests
{
    [TestFixture]
    public class NeuralNetwork_Tests
    {
        private Matrix<double> X;
        private Matrix<double> y;
        private Matrix<double> theta;

        private Matrix<double> Xm;
        private Matrix<double> ym;
        private Matrix<double> bothThetas;

        [TestFixtureSetUp]
        public void TestInitialise()
        {
            CultureSettings.SetNumberDecimalSeparatorToDot();

            int numRows = 20;

            #region initialise X - training data
            X = new DenseMatrix(numRows, 3);
            for (int i = 0; i < numRows; i++)
            {
                X[i, 0] = 1;
            }

            for (int i = 0; i < numRows; i++)
            {
                X[i, 1] = Math.Exp(1) * Math.Sin(i + 1);
            }

            for (int i = 0; i < numRows; i++)
            {
                X[i, 2] = Math.Exp(0.5) * Math.Cos(i + 1);
            }
            #endregion

            #region initialise y - output data

            y = new DenseMatrix(numRows, 1);

            for (int i = 0; i < numRows; i++)
            {
                y[i, 0] = Math.Sin((X[i, 0] + X[i, 1])) > 0 ? 1 : 0;
            }
            #endregion

            #region initialise theta - parameters to find actually from gradient descent

            theta = new DenseMatrix(3, 1);

            //[0.25 0.5 -0.5]
            theta[0, 0] = 0.25;
            theta[1, 0] = 0.5;
            theta[2, 0] = -0.5;

            #endregion

            #region Xm // Xm = reshape(sin(1:32), 16, 2) / 5;

            int numRowsXm = 32;
            var XmData = new double[numRowsXm];
            for (int i = 1; i <= numRowsXm; i++)
            {
                XmData[i - 1] = Math.Sin(i) / 5;
            }
            Xm = new DenseMatrix(16, 2, XmData);

            #endregion

            #region ym // ym = 1 + mod(1:16,4)';

            int numRowsym = 16;
            ym = new DenseMatrix(numRowsym, 1);
            for (int i = 1; i <= numRowsym; i++)
            {
                ym[i - 1, 0] = i % 4 + 1;
            }

            #endregion

            #region // theta1 = sin(reshape(1:2:24, 4, 3));

            var theta1 = new DenseMatrix(4, 3);
            int k = 1;
            for (int j = 0; j < 3; j++)
            {
                for (int i = 0; i < 4; i++)
                {
                    theta1[i, j] = Math.Sin(k);
                    k += 2;
                }
            }

            #endregion

            #region // theta2 = cos(reshape(1:2:40, 4, 5));

            var theta2 = new DenseMatrix(4, 5);
            int t = 1;
            for (int j = 0; j < 5; j++)
            {
                for (int i = 0; i < 4; i++)
                {
                    theta2[i, j] = Math.Cos(t);
                    t += 2;
                }
            }

            #endregion

            #region //both thetas = theta1 and theta2 - in one matrix representation

            var theta1Data = theta1.ToColumnWiseArray();
            var theta2Data = theta2.ToColumnWiseArray();
            var bothThetasData = theta1Data.Concat(theta2Data).ToArray();
            bothThetas = new DenseMatrix(theta1Data.Length + theta2Data.Length, 1, bothThetasData);
     

            #endregion
        }

        private string GetAssemblyPath()
        {
            string codeBase = Assembly.GetExecutingAssembly().CodeBase;
            UriBuilder uri = new UriBuilder(codeBase);
            string path = Uri.UnescapeDataString(uri.Path);
            return Path.GetDirectoryName(path);
        }

        public bool MatricesEqual(Matrix<double> m1, Matrix<double> m2, double difference = 0.00000000001)
        {
            bool result = true;

            for (int i = 0; i < m1.RowCount; i++)
            {
                for (int j = 0; j < m1.ColumnCount; j++)
                {

                    if (Math.Abs(m1[i, j] - m2[i, j]) <= difference)
                    {
                        result &= true;
                    }
                    else
                    {
                        result &= false;
                    }
                }
            }
            return result;
        }

        [Test]
        public void TestSigmoid()
        {
            var sigmoid = NeuralNetwork.Sigmoid(X);

            var path = Path.Combine(GetAssemblyPath(), "..\\..\\TestData\\Sigmoid.txt");
            var matrixToCompare = (DenseMatrix)DelimitedReader.Read<double>(path);

            Assert.IsTrue(MatricesEqual(sigmoid, matrixToCompare));
        }

        [Test]
        public void TestCostFunction()
        {
            var tupleJAndGrad = NeuralNetwork.CostFunction(theta, X, y);

            var pathJ = Path.Combine(GetAssemblyPath(), "..\\..\\TestData\\J.txt");
            var matrixJ = (DenseMatrix)DelimitedReader.Read<double>(pathJ);

            var pathGrad = Path.Combine(GetAssemblyPath(), "..\\..\\TestData\\Grad.txt");
            var matrixGrad = (DenseMatrix)DelimitedReader.Read<double>(pathGrad);

            Assert.IsTrue(Equalities.DoubleEquals(tupleJAndGrad.Item1, matrixJ[0, 0]));
            Assert.IsTrue(MatricesEqual(tupleJAndGrad.Item2, matrixGrad));
        }

        private Tuple<double, Matrix<double>> Equation(Matrix<double> X)
        {
            double x = X[0, 0];
            double y = X[1, 0];

            var f = Math.Pow(x, 2) + Math.Pow(y, 2);
            var dx = 2 * x;
            var dy = 2 * y;

            Matrix<double> df = new DenseMatrix(2, 1);
            df[0, 0] = dx;
            df[1, 0] = dy;

            return Tuple.Create(f, df);
        }

        [Test]
        public void TestGradientDescent()
        {
            Matrix<double> theta = new DenseMatrix(2, 1);
            theta[0, 0] = 1;
            theta[1, 0] = 1;

            double alpha = 0.1;
            int numIterations = 100;

            var result = NeuralNetwork.GradientDescent(Equation, theta, alpha, numIterations);

            var resultTheta = result.Item1;
            Assert.IsTrue(Equalities.DoubleEquals(resultTheta[0, 0], 0, 0.00000001));
            Assert.IsTrue(Equalities.DoubleEquals(resultTheta[1, 0], 0, 0.00000001));

            var resultCost = result.Item2.ToRowWiseArray().LastOrDefault(elem => elem != 0.0d);
            Assert.IsTrue(Equalities.DoubleEquals(resultCost, 0));
        }

        [Test]
        public void TestBackPropagationWithoutRegularisaion()
        {
            NeuralNetwork.CostFunctionWithThetaParameter backProp =
                t =>
                {
                    var backPropagationResult = NeuralNetwork.BackPropagation(t, 2, 4, 4, Xm, ym, 0);
                    return backPropagationResult;
                };

            var resultNumericalGradient = NeuralNetwork.ComputeNumericalGradient(backProp, bothThetas);
            var pathNumericalGradients = Path.Combine(GetAssemblyPath(), "..\\..\\TestData\\NumericalGradientForBackPropagationWithoutRegularisation.txt");
            var matrixNumericalGradients = DelimitedReader.Read<double>(pathNumericalGradients);
            Assert.IsTrue(MatricesEqual(resultNumericalGradient, matrixNumericalGradients));

            var resultBackPropagation = NeuralNetwork.BackPropagation(bothThetas, 2, 4, 4, Xm, ym, 0);
            var pathGradientForBackPropagation = Path.Combine(GetAssemblyPath(), "..\\..\\TestData\\GradientForBackPropagationWithoutRegularisation.txt");
            var matrixGradientForBackPropagation = DelimitedReader.Read<double>(pathGradientForBackPropagation);
            Assert.IsTrue(Equalities.DoubleEquals(resultBackPropagation.Item1, 3.08744915815864));
            
            Assert.IsTrue(MatricesEqual(resultBackPropagation.Item2, resultNumericalGradient, 0.000000001));
        }

        [Test]
        public void TestBackPropagationWithRegularisation()
        {
            NeuralNetwork.CostFunctionWithThetaParameter backProp =
                t =>
                {
                    var backPropagationResult = NeuralNetwork.BackPropagation(t, 2, 4, 4, Xm, ym, 1);
                    return backPropagationResult;
                };

            var resultNumericalGradient = NeuralNetwork.ComputeNumericalGradient(backProp, bothThetas);
            var pathNumericalGradients = Path.Combine(GetAssemblyPath(), "..\\..\\TestData\\NumericalGradients.txt");
            var matrixNumericalGradients = DelimitedReader.Read<double>(pathNumericalGradients);
            Assert.IsTrue(MatricesEqual(resultNumericalGradient, matrixNumericalGradients));

            var resultBackPropagation = NeuralNetwork.BackPropagation(bothThetas, 2, 4, 4, Xm, ym, 1);
            var pathGradientForBackPropagation = Path.Combine(GetAssemblyPath(), "..\\..\\TestData\\GradientForBackPropagation.txt");
            var matrixGradientForBackPropagation = DelimitedReader.Read<double>(pathGradientForBackPropagation);
            Assert.IsTrue(Equalities.DoubleEquals(resultBackPropagation.Item1, 3.46051055642594));
            Assert.IsTrue(MatricesEqual(resultBackPropagation.Item2, matrixGradientForBackPropagation));

            Assert.IsTrue(MatricesEqual(resultBackPropagation.Item2, resultNumericalGradient, 0.000000001));

            var resultGradientDescent = NeuralNetwork.GradientDescent(backProp, bothThetas, 1, 3000);
            var pathResultTheta = Path.Combine(GetAssemblyPath(), "..\\..\\TestData\\ThetaAfterGradientDescentForBackProp.txt");
            var matrixResultTheta = DelimitedReader.Read<double>(pathResultTheta);
            Assert.IsTrue(MatricesEqual(resultGradientDescent.Item1, matrixResultTheta));
            var resultCost = resultGradientDescent.Item2.ToRowWiseArray().LastOrDefault(elem => elem != 0.0d);
            Assert.IsTrue(Equalities.DoubleEquals(resultCost, 2.24934057847533));
        }
    }
}
