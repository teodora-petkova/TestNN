using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CommonLib;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra;
using System.Diagnostics;

namespace TestNeuralNetwork
{

    public class ForwardPropagationResult
    {
        public Matrix<double> a1;
        public Matrix<double> z2;
        public Matrix<double> a2;   
        public Matrix<double> z3;
        public Matrix<double> h;

        public ForwardPropagationResult(Matrix<double> a1,  
            Matrix<double> z2, Matrix<double> a2, Matrix<double> z3, Matrix<double> h)
        {
            this.a1 = a1;
            this.a2 = a2;
            this.h = h;
            this.z2 = z2;
            this.z3 = z3;
        }
    }

    public static class NeuralNetwork
    {
        private static Matrix<double> DoublePowMatrix(double p, Matrix<double> a)
        {
            var b = new DenseMatrix(a.RowCount, a.ColumnCount);

            for (int i = 0; i < a.RowCount; i++)
            {
                for (int j = 0; j < a.ColumnCount; j++)
                {
                    b[i, j] = Math.Pow(p, a[i, j]);
                }
            }

            return b;
        }

        public static Matrix<double> Sigmoid(Matrix<double> Z)
        {
            return (1 / (DoublePowMatrix(Math.Exp(1), (Z * (-1))) + 1));
        }

        public static Matrix<double> SigmoidGradient(Matrix<double> Z)
        {
            return Sigmoid(Z).PointwiseMultiply((1 - Sigmoid(Z)));
        }

        public static Matrix<double> HypothesisFunction(Matrix<double> X, Matrix<double> theta)
        {
            var Z = X * theta;
            return Sigmoid(Z);
        }

        public static Tuple<double, Matrix<double>> CostFunction(Matrix<double> theta, Matrix<double> X, Matrix<double> y, double lambda = 1)
        {
            var h = HypothesisFunction(X, theta);

            int m = y.RowCount;

            var logH = h.PointwiseLog();
            var log1_H = (1 - h).PointwiseLog();
            
            var y1 = (y.Transpose() * logH);
            var y0 = ((1 - y).Transpose() * log1_H);

            //J = (-1/m) * (sum(y' * log(h) + (1 - y)' * log(1-h))) + (lambda/(2*m)) * sum(theta(2:size(theta)).^2)';
            var theta0 = theta.SubMatrix(1, theta.RowCount - 1, 0, theta.ColumnCount);

            var J = (y1 + y0).ColumnSums() * (double)(-1d / m) + ((theta0.PointwisePower(2)).ColumnSums()) * (double)(lambda / (2d * m));

            // x_o = X(:,1);
            // x_rest = X(:, 2:end);

            // grad(1) = (1/m) * sum((h - y)'*x_o, 1);
            // grad(2:end) = (1/m) * sum((h - y)'*x_rest, 1)' + (lambda/m) * theta(2:end);

            var x0 = X.SubMatrix(0, X.RowCount, 0, 1);
            var xRest = X.SubMatrix(0, X.RowCount, 1, X.ColumnCount-1);

            var transposedh_y = (h - y).Transpose();
            var h_y_x0 = transposedh_y * x0;
            var h_y_xRest = transposedh_y * xRest;

            var sum1 = new DenseMatrix(1, h_y_x0.ColumnCount, (h_y_x0.ColumnSums().ToArray()));
            var sum2 = new DenseMatrix(1, h_y_xRest.ColumnCount, (h_y_xRest.ColumnSums().ToArray()));

            var grad1 = ((double)(1d / m)) * (sum1);
            var grad2 = ((double)(1d / m)) * (sum2) + (double)(lambda / m) * theta0.Transpose();

            var grad = grad1.Append(grad2);

            var result = Tuple.Create(J[0], grad);

            return result;
        }

        // Gradient Descent
        public static Tuple<Matrix<double>, Matrix<double>> GradientDescent(CostFunctionWithThetaParameter func,
            Matrix<double> theta, double alpha, int numberIterations)
        {
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();
            
            Matrix<double> JHistory = new DenseMatrix(numberIterations, 1);
            var alphaHistory = new double[numberIterations];
            for (int i = 0; i < numberIterations; i++)
            {
                var res = func(theta);

                var h = res.Item1;
                var grad = res.Item2;
                JHistory[i, 0] = h;

                //double[] temp = new double[theta.Rows];
                //for (int j = 0; j < theta.Rows; j++)
                //{
                //    temp[j] = theta[j, 0] - (grad[j, 0] * alpha);
                //}

                //theta = new Matrix(theta.Rows, 1);
                //for (int j = 0; j < theta.Rows; j++)
                //{
                //    theta[j, 0] = temp[j];
                //}

                theta = theta - grad * alpha;

                // "bold driver" - if we decrease the cost function, increase the learning rate by 5% but
                // in case when we increase the cost function, decrease the learning rate by 50%
                if (i > 0)
                {
                    if (JHistory[i, 0] < JHistory[i - 1, 0])
                    {
                        alpha += 0.05 * alpha;
                    }
                    else
                    {
                        alpha -= 0.5 * alpha;
                    }
                }

                alphaHistory[i] = alpha;

                if (i > 0 && JHistory[i, 0] < JHistory[i - 1, 0] &&
                    Equalities.DoubleEquals(JHistory[i, 0], JHistory[i - 1, 0]))
                {
                    break;
                }   
            }

            var m = new DenseMatrix(numberIterations, 1, alphaHistory);
            
            stopWatch.Stop();
            // Get the elapsed time as a TimeSpan value.
            TimeSpan ts = stopWatch.Elapsed;

            // Format and display the TimeSpan value. 
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                ts.Hours, ts.Minutes, ts.Seconds, ts.Milliseconds / 10);
            Console.WriteLine("RunTime " + elapsedTime);

            return Tuple.Create(theta, JHistory);
        }

        public delegate Tuple<double, Matrix<double>> CostFunctionWithThetaParameter(Matrix<double> m);

        public static Matrix<double> ComputeNumericalGradient(CostFunctionWithThetaParameter J, Matrix<double> theta)
        {
            double epsilon = 0.0001;

            Matrix<double> numericalGradient = new DenseMatrix(theta.RowCount, 1);
            var perturbations = new DenseMatrix(theta.RowCount, 1); // смущения ;))

            for (int p = 0; p < theta.RowCount; p++)
            {
                perturbations[p, 0] = epsilon;

                double loss1 = J(theta + perturbations).Item1;
                double loss2 = J(theta - perturbations).Item1;

                numericalGradient[p, 0] = ((loss1 - loss2) / (double)(2d * epsilon));

                perturbations[p, 0] = 0;
            }

            return numericalGradient;
        }

        public static List<Matrix<double>> UnpackThetas(Matrix<double> theta,
            int inputLayerSize, int hiddenLayerSize, int numberLabels)
        {
            var theta1 = MatriceSubRowMatrixAndReshape(theta, hiddenLayerSize, inputLayerSize + 1,
                   0, hiddenLayerSize * (inputLayerSize + 1));
            var theta2 = MatriceSubRowMatrixAndReshape(theta, numberLabels, hiddenLayerSize + 1,
                hiddenLayerSize * (inputLayerSize + 1), theta.RowCount - hiddenLayerSize * (inputLayerSize + 1));

            return new List<Matrix<double>>() { theta1, theta2 };
        }

        // Forward Propagation
        public static ForwardPropagationResult ForwardPropagation(Matrix<double> X, 
            IEnumerable<Matrix<double>> thetas, int inputLayerSize, int hiddenLayerSize, int numberLabels)
        {
            var theta1 = thetas.ElementAt(0);
            var theta2 = thetas.ElementAt(1);

            int m = X.RowCount;
           
            /// forward propagation   
            var a1 = X;
            var onesa1 = GetMatrixColumnOfOnes(a1.RowCount); 
            var a11 = onesa1.Append(a1);

            var z2 = (a11 * theta1.Transpose());
            var a2 = Sigmoid(z2);
            var onesa2 = GetMatrixColumnOfOnes(a2.RowCount); 
            var a12 = onesa2.Append(a2);

            var z3 = (a12 * theta2.Transpose());
            var a3 = Sigmoid(z3);

            var h = a3;

            return new ForwardPropagationResult(a11, z2, a12, z3, h);
        }

        public static void Populate<T>(this T[] arr, T value)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                arr[i] = value;
            }
        }

        public static Matrix<double> GetOutputVectors(Matrix<double> yClasses, int numberLabels)
        {
            int m = yClasses.RowCount;
            
            var y = new DenseMatrix(m, numberLabels);

            for(int i=0; i<m; i++)
            {
                y[i, (int)yClasses[i, 0]-1] = 1;
            }

            return y;
        }

        public static Tuple<double, Matrix<double>> BackPropagation(Matrix<double> theta,
            int inputLayerSize, int hiddenLayerSize, int numberLabels,
            Matrix<double> X, Matrix<double> y, double lambda)
        {
            int m = X.RowCount;
            var thetas = UnpackThetas(theta, inputLayerSize, hiddenLayerSize, numberLabels);
            var theta1 = thetas.ElementAt(0);
            var theta2 = thetas.ElementAt(1);

            /// feed forward
            var feedForwardResult = ForwardPropagation(X, thetas, inputLayerSize, hiddenLayerSize, numberLabels);

            var newY = GetOutputVectors(y, feedForwardResult.h.ColumnCount);

            /// back propagation
            double J = BackPropagationCostFunction(feedForwardResult.h, newY,
                inputLayerSize, hiddenLayerSize, numberLabels,
                theta1, theta2, lambda, m);

            var gradient = GetBackPropagationGradient(feedForwardResult, newY,
                inputLayerSize, hiddenLayerSize, numberLabels,
                theta1, theta2, lambda, m);
           
            return Tuple.Create(J, gradient);
        }

        private static Matrix<double> GetBackPropagationGradient(ForwardPropagationResult feedForward,
            Matrix<double> newY, int inputLayerSize, int hiddenLayerSize, int numberLabels,
            Matrix<double> theta1, Matrix<double> theta2, double lambda, int m)
        {
            var delta3 = feedForward.h - newY;
            var r2 = (theta2.SubMatrix(0, theta2.RowCount, 1, theta2.ColumnCount - 1)).Transpose() * delta3.Transpose();

            var delta2 = r2.Transpose().PointwiseMultiply(SigmoidGradient(feedForward.z2));

            var theta1_grad = delta2.Transpose() * feedForward.a1;
            var theta2_grad = delta3.Transpose() * feedForward.a2;

            var t1 = (lambda * theta1) / m;
            var zerosFort1 = GetArrayOfNumber(0, t1.RowCount);
            t1.SetColumn(0, zerosFort1);

            var t2 = (lambda * theta2) / m;
            var zerosFort2 = GetArrayOfNumber(0, t2.RowCount);
            t2.SetColumn(0, zerosFort2);

            theta1_grad = (theta1_grad / m) + t1;
            theta2_grad = (theta2_grad / m) + t2;

            var theta1_grad_arr = theta1_grad.ToColumnWiseArray();
            var theta2_grad_arr = theta2_grad.ToColumnWiseArray();

            var grad = theta1_grad_arr.Concat(theta2_grad_arr).ToArray();
            Matrix<double> gradMatrix = new DenseMatrix(grad.Count(), 1, grad);

            return gradMatrix;
        }

        private static double[] GetArrayOfNumber(double number, int size)
        {
            var numbers = new double[size];
            numbers.Populate(number);
            return numbers;
        }

        private static Matrix<double> GetMatrixColumnOfOnes(int rowCount)
        {
            var onesMatrix = new DenseMatrix(rowCount, 1, GetArrayOfNumber(1, rowCount));

            return onesMatrix;
        }

        private static Matrix<double> MatriceSubRowMatrixAndReshape(Matrix<double> matrix, int rowCount, int columnCount, int from, int to)
        {
            var subMatrix = matrix.SubMatrix(from, to, 0, matrix.ColumnCount);
            var dataMatrix = subMatrix.ToRowWiseArray();
            var newMatrix = new DenseMatrix(rowCount, columnCount, dataMatrix);

            return newMatrix;
        }

        private static double BackPropagationCostFunction(Matrix<double> h, Matrix<double> y,
            int inputLayerSize, int hiddenLayerSize, int numberLabels,
            Matrix<double> theta1, Matrix<double> theta2, double lambda, int numberOfTestSamples)
        {
            int m = numberOfTestSamples;
         
            double cost = 0;
            for (int i = 0; i < m; i++)
            {
                for (int l = 0; l < numberLabels; l++)
                {
                    var partA = y[i, l] * Math.Log(h[i, l]);
                    var partB = (1 - y[i, l]) * Math.Log(1 - h[i, l]);
                    cost = cost + partA + partB;
                }
            }
            double J = cost * (double)(-1 / (double)m);

            double toAdd = GetRegularisationTerm(inputLayerSize, hiddenLayerSize, numberLabels,
                theta1, theta2, lambda, m);
            
            J = J + toAdd;

            return J;
        }

        private static double GetRegularisationTerm(int inputLayerSize, int hiddenLayerSize, int numberLabels,
            Matrix<double> theta1, Matrix<double> theta2, double lambda, double numberTestSamples)
        {
            var m = numberTestSamples;

            //var zeros1 = new Matrix(theta1.Rows, 1);
            //zeros1.Zero();
            //var zeros2 = new Matrix(theta2.Rows, 1);
            //zeros2.Zero();

            //var thetaNew1 = Matrix.ConcatMatrices(zeros1, theta1.SubMatrix(1, theta1.Columns));
            //var thetaNew2 = Matrix.ConcatMatrices(zeros2, theta2.SubMatrix(1, theta2.Columns));

            double thetaSumA = 0;
            for (int j = 0; j < hiddenLayerSize; j++)
            {
                for (int k = 1; k < inputLayerSize + 1; k++) // without first column - bias term!
                {
                    thetaSumA += Math.Pow(theta1[j, k], 2);
                }
            }

            double thetaSumB = 0;
            for (int j = 0; j < numberLabels; j++)
            {
                for (int k = 1; k < hiddenLayerSize + 1; k++) // without first column - bias term!
                {
                    thetaSumB += Math.Pow(theta2[j, k], 2);
                }
            }

            var toAdd = ((double)(lambda / ((double)(2 * m)))) * (thetaSumA + thetaSumB);

            return toAdd;
        }
    }
}
