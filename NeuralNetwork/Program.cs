using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CommonLib;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Data.Text;

namespace TestNeuralNetwork
{
    public class Program
    {
        private static Random random = new Random();
        
        private static double GetRandom(double min, double max)
        {
            // initialise weights randomly so that we break the symmetry while
            // training the neural network
            return (random.NextDouble()*(max - min) + min);
        }

        static void Main(string[] args)
        {
            CultureSettings.SetNumberDecimalSeparatorToDot();

            int numRows = 20;

            #region initialise X - training data
            var X = new DenseMatrix(numRows, 3);
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

            var y = new DenseMatrix(numRows, 1);

            for (int i = 0; i < numRows; i++)
            {
                y[i, 0] = Math.Sin((X[i, 0] + X[i, 1])) > 0 ? 1 : 0;
            }
            #endregion

            var result1 = NeuralNetwork.Sigmoid(X);

            #region initialise theta - parameters to find actually from gradient descent

            var theta = new DenseMatrix(3, 1);

            //[0.25 0.5 -0.5]
            theta[0, 0] = 0.25;
            theta[1, 0] = 0.5;
            theta[2, 0] = -0.5;

            #endregion

            var result2 = NeuralNetwork.CostFunction(theta, X, y);

            //var result3 = NeuralNetwork.ForwardPropagation(X, new List<Matrixdouble>>() { theta });

            #region X1 // X1 = [ones(20,1) (exp(1) + exp(2) * (0.1:0.1:2))'];

            var X1 = new DenseMatrix(numRows, 2);
            for (int i = 0; i < numRows; i++)
            {
                X1[i, 0] = 1;
            }

            double d = 0.1d;
            for (int i = 0; i < numRows; i++)
            {
                X1[i, 1] = Math.Exp(1) + Math.Exp(2) * d;
                d += 0.1d;
            }

            #endregion

            #region Y1 // Y1 = X1(:,2) + sin(X1(:,1)) + cos(X1(:,2));

            var Y1 = new DenseMatrix(numRows, 1);

            for (int i = 0; i < numRows; i++)
            {
                Y1[i, 0] = X1[i, 1] + Math.Sin(X1[i, 0]) + Math.Cos(X1[i, 1]);
            }

            #endregion

            #region X2 // X2 = [X1 X1(:,2).^0.5 X1(:,2).^0.25];

            var X2 = new DenseMatrix(numRows, 4);

            for (int i = 0; i < numRows; i++)
            {
                X2[i, 0] = X1[i, 0];
            }

            for (int i = 0; i < numRows; i++)
            {
                X2[i, 1] = X1[i, 1];
            }

            for (int i = 0; i < numRows; i++)
            {
                X2[i, 2] = Math.Pow(X1[i, 1], 0.5);
            }

            for (int i = 0; i < numRows; i++)
            {
                X2[i, 3] = Math.Pow(X1[i, 1], 0.25);
            }

            #endregion

            #region Y2 // Y2 = Y1.^0.5 + Y1;

            var Y2 = new DenseMatrix(numRows, 1);

            for (int i = 0; i < numRows; i++)
            {
                Y2[i, 0] = Math.Pow(Y1[i, 0], 0.5) + Y1[i, 0];
            }

            #endregion

            #region Xm // Xm = reshape(sin(1:32), 16, 2) / 5;
            
            int numRowsXm = 32;
            var XmData = new double[numRowsXm];
            for (int i = 1; i <= numRowsXm; i++)
            {
                XmData[i - 1] = Math.Sin(i) / 5;
            }
            var Xm = new DenseMatrix(16, 2, XmData);

            #endregion

            #region ym // ym = 1 + mod(1:16,4)';
            
            int numRowsym = 16;
            var ym = new DenseMatrix(numRowsym, 1);
            for (int i = 1; i <= numRowsym; i++)
            {
                ym[i-1, 0] = i%4 + 1;
            }
            
            #endregion

            #region // theta1 = sin(reshape(1:2:24, 4, 3));

            var theta1 = new DenseMatrix(4, 3);
            int k = 1;             
            for(int j = 0; j < 3; j++)
            {
                for (int i = 0; i < 4; i++)
                {
                    theta1[i, j]= Math.Sin(k);
                    k+=2;
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
            
            var theta1Data = theta1.ToRowWiseArray();
            var theta2Data = theta2.ToRowWiseArray();
            var bothThetasData = theta1Data.Concat(theta2Data).ToArray();
            var bothThetas = new DenseMatrix(theta1Data.Length + theta2Data.Length, 1, bothThetasData);
            
            #endregion

            //var x_data = (DenseMatrix)DelimitedReader.Read<double>("D:/test/x_data.txt");
            //var y_data = (DenseMatrix)DelimitedReader.Read<double>("D:/test/y_data.txt");

            //NeuralNetwork.CostFunctionWithThetaParameter backProp =
            //    tt =>
            //    {
            //        var backPropagationResult = NeuralNetwork.BackPropagation(tt, 400, 25, 10, x_data, y_data, 1);
            //        return backPropagationResult;
            //    };


            //Random random = new Random();
            //var data1 = new double[25 * 401];
            //for (int i = 0; i < 25 * 401; i++)
            //{
            //    data1[i] = GetRandom(400, 25);
            //}

            //var data2 = new double[26 * 10];
            //for (int i = 0; i < 26 * 10; i++)
            //{
            //    data2[i] = GetRandom(25, 10);
            //}

            //var theta1_data = new DenseMatrix(25, 401, data1);
            //var theta2_data = new DenseMatrix(10, 26, data2);

            //DenseMatrix bothThetas_data = new DenseMatrix(data1.Length + data2.Length, 1, data1.Concat(data2).ToArray());

            //var result6 = NeuralNetwork.GradientDescent(backProp, bothThetas_data, 1, 300);
            //DelimitedWriter.Write("D:/test/thetaGradient.txt", result6.Item1, "thetaGradient");
            //DelimitedWriter.Write("D:/test/JHistory.txt", result6.Item2, "JHistory");

            //var resultTheta = result6.Item1;

            ////var result4 = NeuralNetwork.ComputeNumericalGradient(backProp, resultTheta);
            ////result4.Save("D:/test/numericalGradients.txt");

            //var result5 = NeuralNetwork.BackPropagation(resultTheta, 400, 25, 10, x_data, y_data, 1);
            //var JJ = result5.Item1;
            //DelimitedWriter.Write("D:/test/gradBackPropagation.txt", result5.Item2, "gradBackPropagation");

            //var thetas = (DenseMatrix)DelimitedReader.Read<double>("D:/test/thetaGradient.txt");
            //var thetasList = NeuralNetwork.UnpackThetas(thetas, 400, 45, 10);
            //var result8 = NeuralNetwork.ForwardPropagation(x_data, thetasList, 400, 25, 10);

            //var predictions = new DenseMatrix(result8.h.RowCount, 1);
            //for (int i = 0; i < result8.h.ColumnCount; i++)
            //{
            //    var max = result8.h[i, 0];
            //    var predictedClass = 0;
            //    for (int j = 0; j < result8.h.ColumnCount; j++)
            //    {
            //        if (result8.h[i, j] > max)
            //        {
            //            max = result8.h[i, j];
            //            predictedClass = j;
            //        }
            //    }
            //    predictions[i, 0] = predictedClass;
            //}
            //DelimitedWriter.Write("D:/test/predictions.txt", predictions, "");
        }
    }
}
