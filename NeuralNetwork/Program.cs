using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CommonLib;

namespace TestNeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            CultureSettings.SetNumberDecimalSeparatorToDot();

            int numRows = 20;

            #region initialise X - training data
            var X = new Matrix(numRows, 3);
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

            var y = new Matrix(numRows, 1);

            for (int i = 0; i < numRows; i++)
            {
                y[i, 0] = Math.Sin((X[i, 0] + X[i, 1])) > 0 ? 1 : 0;
            }
            #endregion

            var result1 = NeuralNetwork.Sigmoid(X);

            #region initialise theta - parameters to find actually from gradient descent

            var theta = new Matrix(3, 1);

            //[0.25 0.5 -0.5]
            theta[0, 0] = 0.25;
            theta[1, 0] = 0.5;
            theta[2, 0] = -0.5;

            #endregion

            var result2 = NeuralNetwork.CostFunction(theta, X, y);

            var result3 = NeuralNetwork.ForwardPropagation(X, new List<Matrix>() { theta });

            #region X1 // X1 = [ones(20,1) (exp(1) + exp(2) * (0.1:0.1:2))'];

            var X1 = new Matrix(numRows, 2);
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

            var Y1 = new Matrix(numRows, 1);

            for (int i = 0; i < numRows; i++)
            {
                Y1[i, 0] = X1[i, 1] + Math.Sin(X1[i, 0]) + Math.Cos(X1[i, 1]);
            }

            #endregion

            #region X2 // X2 = [X1 X1(:,2).^0.5 X1(:,2).^0.25];

            var X2 = new Matrix(numRows, 4);

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

            var Y2 = new Matrix(numRows, 1);

            for (int i = 0; i < numRows; i++)
            {
                Y2[i, 0] = Math.Pow(Y1[i, 0], 0.5) + Y1[i, 0];
            }

            #endregion

            #region Xm // Xm = reshape(sin(1:32), 16, 2) / 5;
            
            int numRowsXm = 32;
            var Xm = new Matrix(numRowsXm, 1);
            for (int i = 1; i <= numRowsXm; i++)
            {
                Xm[i-1, 0] = Math.Sin(i)/ 5;
            }
            Xm.Reshape(16, 2);

            #endregion

            #region ym // ym = 1 + mod(1:16,4)';
            
            int numRowsym = 16;
            var ym = new Matrix(numRowsym, 1);
            for (int i = 1; i <= numRowsym; i++)
            {
                ym[i-1, 0] = i%4 + 1;
            }
            
            #endregion

            #region // theta1 = sin(reshape(1:2:24, 4, 3));

            var theta1 = new Matrix(4, 3);
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
            Matrix bothThetas = Matrix.ConcatMatrices(theta1, theta2, false);
            
            #endregion

            NeuralNetwork.CostFunctionWithThetaParameter backProp =
                tt =>
                {
                    var backPropagationResult = NeuralNetwork.BackPropagation(tt, 2, 4, 4, Xm, ym, 1);
                    return backPropagationResult;
                };

            var result4 = NeuralNetwork.ComputeNumericalGradient(backProp, bothThetas);
            //result4.Save("D:/test/numericalGradients.txt");

            var result5 = NeuralNetwork.BackPropagation(bothThetas, 2, 4, 4, Xm, ym, 1);
            var JJ = result5.Item1;
            //result5.Item2.Save("D:/test/gradBackPropagation.txt");

            //var result6 = NeuralNetwork.GradientDescent(backProp, bothThetas_data, 1, 3);
            //result6.Item1.Save("D:/test/thetaGradient.txt");
            //result6.Item2.Save("D:/test/JHistory.txt");

        }
    }
}
