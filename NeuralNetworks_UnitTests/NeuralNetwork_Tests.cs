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
        private Matrix<double> theta1NN;
        private Matrix<double> theta2NN;

        [TestFixtureSetUp]
        public void TestInitialise()
        {
            Utils.SetNumberDecimalSeparatorToDotInCultureSettings();

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
            theta1NN = new DenseMatrix(4, 3, theta1Data);
            theta2NN = new DenseMatrix(4, 5, theta2Data);

            #endregion
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

            var path = Path.Combine(Utils.GetAssemblyPath(), "..\\..\\TestData\\Sigmoid.txt");
            var matrixToCompare = (DenseMatrix)DelimitedReader.Read<double>(path);

            Assert.IsTrue(MatricesEqual(sigmoid, matrixToCompare));
        }

        [Test]
        public void TestCostFunction()
        {
            var tupleJAndGrad = NeuralNetwork.CostFunction(theta, X, y);

            var pathJ = Path.Combine(Utils.GetAssemblyPath(), "..\\..\\TestData\\J.txt");
            var matrixJ = (DenseMatrix)DelimitedReader.Read<double>(pathJ);

            var pathGrad = Path.Combine(Utils.GetAssemblyPath(), "..\\..\\TestData\\Grad.txt");
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
        public void TestForwardPropagation()
        {

            var resultFeedForward = NeuralNetwork.ForwardPropagation(Xm, new List<Matrix<double>>() { theta1NN, theta2NN });

            var h = resultFeedForward.OutputActivation;
            var a1 = resultFeedForward.NodesActivations.ElementAt(0);
            var a2 = resultFeedForward.NodesActivations.ElementAt(1);
            var z2 = resultFeedForward.NodesOutputs.ElementAt(0);
            var z3 = resultFeedForward.NodesOutputs.ElementAt(1);

            var pathH = Path.Combine(Utils.GetAssemblyPath(), "..\\..\\TestData\\FeedForward_h.txt");
            var patha1 = Path.Combine(Utils.GetAssemblyPath(), "..\\..\\TestData\\FeedForward_a1.txt");
            var patha2 = Path.Combine(Utils.GetAssemblyPath(), "..\\..\\TestData\\FeedForward_a2.txt");
            var pathz2 = Path.Combine(Utils.GetAssemblyPath(), "..\\..\\TestData\\FeedForward_z2.txt");
            var pathz3 = Path.Combine(Utils.GetAssemblyPath(), "..\\..\\TestData\\FeedForward_z3.txt");

            var wantedH = DelimitedReader.Read<double>(pathH);
            var wanteda1 = DelimitedReader.Read<double>(patha1);
            var wanteda2 = DelimitedReader.Read<double>(patha2);
            var wantedz2 = DelimitedReader.Read<double>(pathz2);
            var wantedz3 = DelimitedReader.Read<double>(pathz3);

            Assert.IsTrue(MatricesEqual(h, wantedH));
            Assert.IsTrue(MatricesEqual(a1, wanteda1));
            Assert.IsTrue(MatricesEqual(a2, wanteda2));
            Assert.IsTrue(MatricesEqual(z2, wantedz2));
            Assert.IsTrue(MatricesEqual(z3, wantedz3));  

        }

        [Test]
        public void TestBackPropagationWithoutRegularisaion()
        {
            NeuralNetwork.CostFunctionWithThetaParameter backProp =
                t =>
                {
                    var backPropagationResult = NeuralNetwork.BackPropagation(Xm, ym, t, new List<int>() { 4 }, 4, 0);
                    return backPropagationResult;
                };

            var bothThetas = NeuralNetwork.PackThetas(new List<Matrix<double>>() { theta1NN, theta2NN });
            var resultNumericalGradient = NeuralNetwork.ComputeNumericalGradient(backProp, bothThetas);
            var pathNumericalGradients = Path.Combine(Utils.GetAssemblyPath(), "..\\..\\TestData\\NumericalGradientForBackPropagationWithoutRegularisation.txt");
            var matrixNumericalGradients = DelimitedReader.Read<double>(pathNumericalGradients);
            Assert.IsTrue(MatricesEqual(resultNumericalGradient, matrixNumericalGradients));

            var resultBackPropagation = NeuralNetwork.BackPropagation(Xm, ym, bothThetas, new List<int>() { 4 }, 4, 0);
            var pathGradientForBackPropagation = Path.Combine(Utils.GetAssemblyPath(), "..\\..\\TestData\\GradientForBackPropagationWithoutRegularisation.txt");
            var matrixGradientForBackPropagation = DelimitedReader.Read<double>(pathGradientForBackPropagation);
            Assert.IsTrue(Equalities.DoubleEquals(resultBackPropagation.Item1, 3.08744915815864));
            Assert.IsTrue(MatricesEqual(resultBackPropagation.Item2, matrixGradientForBackPropagation));  
            Assert.IsTrue(MatricesEqual(resultBackPropagation.Item2, resultNumericalGradient, 0.000000001));
        }

        [Test]
        public void TestBackPropagationWithRegularisation()
        {
            NeuralNetwork.CostFunctionWithThetaParameter backProp =
                t =>
                {
                    var backPropagationResult = NeuralNetwork.BackPropagation(Xm, ym, t, new List<int>() { 4 }, 4, 1);
                    return backPropagationResult;
                };

            var bothThetas = NeuralNetwork.PackThetas(new List<Matrix<double>>() { theta1NN, theta2NN });
            var resultNumericalGradient = NeuralNetwork.ComputeNumericalGradient(backProp, bothThetas);
            var pathNumericalGradients = Path.Combine(Utils.GetAssemblyPath(), "..\\..\\TestData\\NumericalGradients.txt");
            var matrixNumericalGradients = DelimitedReader.Read<double>(pathNumericalGradients);
            Assert.IsTrue(MatricesEqual(resultNumericalGradient, matrixNumericalGradients));

            var resultBackPropagation = NeuralNetwork.BackPropagation(Xm, ym, bothThetas, new List<int>() { 4 }, 4, 1);
            var pathGradientForBackPropagation = Path.Combine(Utils.GetAssemblyPath(), "..\\..\\TestData\\GradientForBackPropagation.txt");
            var matrixGradientForBackPropagation = DelimitedReader.Read<double>(pathGradientForBackPropagation);
            Assert.IsTrue(Equalities.DoubleEquals(resultBackPropagation.Item1, 3.46051055642594));
            Assert.IsTrue(MatricesEqual(resultBackPropagation.Item2, matrixGradientForBackPropagation));

            Assert.IsTrue(MatricesEqual(resultBackPropagation.Item2, resultNumericalGradient, 0.000000001));

            var resultGradientDescent = NeuralNetwork.GradientDescent(backProp, bothThetas, 1, 3000);
            var pathResultTheta = Path.Combine(Utils.GetAssemblyPath(), "..\\..\\TestData\\ThetaAfterGradientDescentForBackProp.txt");
            var matrixResultTheta = DelimitedReader.Read<double>(pathResultTheta);
            Assert.IsTrue(MatricesEqual(resultGradientDescent.Item1, matrixResultTheta));
            var resultCost = resultGradientDescent.Item2.ToRowWiseArray().LastOrDefault(elem => elem != 0.0d);
            Assert.IsTrue(Equalities.DoubleEquals(resultCost, 2.2493405784756875));
        }
    }
}
