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

namespace NeuralNetworkTests
{
    [TestFixture]
    public class NeuralNetwork_Tests
    {
        private Matrix X;
        private Matrix y;
        private Matrix theta;

        private Matrix Xm;
        private Matrix ym;
        private Matrix bothThetas;

        [TestFixtureSetUp]
        public void TestInitialise()
        {
            CultureSettings.SetNumberDecimalSeparatorToDot();

            int numRows = 20;

            #region initialise X - training data
            X = new Matrix(numRows, 3);
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

            y = new Matrix(numRows, 1);

            for (int i = 0; i < numRows; i++)
            {
                y[i, 0] = Math.Sin((X[i, 0] + X[i, 1])) > 0 ? 1 : 0;
            }
            #endregion

            #region initialise theta - parameters to find actually from gradient descent

            theta = new Matrix(3, 1);

            //[0.25 0.5 -0.5]
            theta[0, 0] = 0.25;
            theta[1, 0] = 0.5;
            theta[2, 0] = -0.5;

            #endregion

            #region Xm // Xm = reshape(sin(1:32), 16, 2) / 5;

            int numRowsXm = 32;
            Xm = new Matrix(numRowsXm, 1);
            for (int i = 1; i <= numRowsXm; i++)
            {
                Xm[i - 1, 0] = Math.Sin(i) / 5;
            }
            Xm.Reshape(16, 2);

            #endregion

            #region ym // ym = 1 + mod(1:16,4)';

            int numRowsym = 16;
            ym = new Matrix(numRowsym, 1);
            for (int i = 1; i <= numRowsym; i++)
            {
                ym[i - 1, 0] = i % 4 + 1;
            }

            #endregion

            #region // theta1 = sin(reshape(1:2:24, 4, 3));

            var theta1 = new Matrix(4, 3);
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

            var theta2 = new Matrix(4, 5);
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

            theta1.Reshape(theta1.Rows * theta1.Columns, 1);
            theta2.Reshape(theta2.Rows * theta2.Columns, 1);
            bothThetas = Matrix.ConcatMatrices(theta1, theta2, false);

            #endregion
        }

        private string GetAssemblyPath()
        {
            string codeBase = Assembly.GetExecutingAssembly().CodeBase;
            UriBuilder uri = new UriBuilder(codeBase);
            string path = Uri.UnescapeDataString(uri.Path);
            return Path.GetDirectoryName(path);
        }

        [Test]
        public void TestSigmoid()
        {
            var sigmoid = NeuralNetwork.Sigmoid(X);

            var path = Path.Combine(GetAssemblyPath(), "..\\..\\TestData\\Sigmoid.txt");
            var matrixToCompare = Matrix.LoadMatrixFromFile(path);

            Assert.IsTrue(sigmoid.Equals(matrixToCompare));
        }

        [Test]
        public void TestCostFunction()
        {
            var tupleJAndGrad = NeuralNetwork.CostFunction(theta, X, y);

            var pathJ = Path.Combine(GetAssemblyPath(), "..\\..\\TestData\\J.txt");
            var matrixJ = Matrix.LoadMatrixFromFile(pathJ);

            var pathGrad = Path.Combine(GetAssemblyPath(), "..\\..\\TestData\\Grad.txt");
            var matrixGrad = Matrix.LoadMatrixFromFile(pathGrad);

            Assert.IsTrue(Equalities.DoubleEquals(tupleJAndGrad.Item1, matrixJ[0, 0]));
            Assert.IsTrue(tupleJAndGrad.Item2.Equals(matrixGrad));
        }

        private Tuple<double, Matrix> Equation(Matrix X)
        {
            double x = X[0, 0];
            double y = X[1, 0];

            var f = Math.Pow(x, 2) + Math.Pow(y, 2);
            var dx = 2 * x;
            var dy = 2 * y;

            var df = new Matrix(2, 1);
            df[0, 0] = dx;
            df[1, 0] = dy;

            return Tuple.Create(f, df);
        }

        [Test]
        public void TestGradientDescent()
        {
            var theta = new Matrix(2, 1);
            theta[0, 0] = 1;
            theta[1, 0] = 1;

            double alpha = 0.1;
            int numIterations = 100;

            var result = NeuralNetwork.GradientDescent(Equation, theta, alpha, numIterations);

            var resultTheta = result.Item1;
            Assert.IsTrue(Equalities.DoubleEquals(resultTheta[0, 0], 0, 0.00000001));
            Assert.IsTrue(Equalities.DoubleEquals(resultTheta[1, 0], 0, 0.00000001));

            var resultCost = result.Item2.GetSingleArrayAsRows().LastOrDefault(elem => elem != 0.0d);
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
            //var pathNumericalGradients = Path.Combine(GetAssemblyPath(), "..\\..\\TestData\\NumericalGradients.txt");
            //var matrixNumericalGradients = Matrix.LoadMatrixFromFile(pathNumericalGradients);
            //Assert.IsTrue(resultNumericalGradient.Equals(matrixNumericalGradients));

            var resultBackPropagation = NeuralNetwork.BackPropagation(bothThetas, 2, 4, 4, Xm, ym, 0);
            //var pathGradientForBackPropagation = Path.Combine(GetAssemblyPath(), "..\\..\\TestData\\GradientForBackPropagation.txt");
            //var matrixGradientForBackPropagation = Matrix.LoadMatrixFromFile(pathGradientForBackPropagation);
            //Assert.IsTrue(Equalities.DoubleEquals(resultBackPropagation.Item1, 3.46051055642594));
            Assert.IsTrue(resultBackPropagation.Item2.Equals(resultNumericalGradient, 0.000000001));
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
            var matrixNumericalGradients = Matrix.LoadMatrixFromFile(pathNumericalGradients);
            Assert.IsTrue(resultNumericalGradient.Equals(matrixNumericalGradients));

            var resultBackPropagation = NeuralNetwork.BackPropagation(bothThetas, 2, 4, 4, Xm, ym, 1);
            var pathGradientForBackPropagation = Path.Combine(GetAssemblyPath(), "..\\..\\TestData\\GradientForBackPropagation.txt");
            var matrixGradientForBackPropagation = Matrix.LoadMatrixFromFile(pathGradientForBackPropagation);
            Assert.IsTrue(Equalities.DoubleEquals(resultBackPropagation.Item1, 3.46051055642594));
            Assert.IsTrue(resultBackPropagation.Item2.Equals(matrixGradientForBackPropagation));

            Assert.IsTrue(resultBackPropagation.Item2.Equals(resultNumericalGradient, 0.000000001));

            var resultGradientDescent = NeuralNetwork.GradientDescent(backProp, bothThetas, 1, 3000);
            var pathResultTheta = Path.Combine(GetAssemblyPath(), "..\\..\\TestData\\ThetaAfterGradientDescentForBackProp.txt");
            var matrixResultTheta = Matrix.LoadMatrixFromFile(pathResultTheta);
            Assert.IsTrue(resultGradientDescent.Item1.Equals(matrixResultTheta));
            var resultCost = resultGradientDescent.Item2.GetSingleArrayAsRows().LastOrDefault(elem => elem != 0.0d);
            Assert.IsTrue(Equalities.DoubleEquals(resultCost, 2.24934057847533));
        }
    }
}
