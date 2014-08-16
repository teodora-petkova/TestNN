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
        public IEnumerable<Matrix<double>> NodesActivations = new List<Matrix<double>>();
        public IEnumerable<Matrix<double>> NodesOutputs = new List<Matrix<double>>();
        public Matrix<double> OutputActivation;
        public ForwardPropagationResult(Matrix<double> h,
            IEnumerable<Matrix<double>> a,
            IEnumerable<Matrix<double>> z)
        {
            this.OutputActivation = h;
            this.NodesActivations = a;
            this.NodesOutputs = z;
        }
    }

    public static class NeuralNetwork
    {
        private static Random random = new Random();

        private static double GetRandom(double min, double max, double epsilon = 0.12)
        {
            // initialise weights randomly in order to break the symmetry while
            // training the neural network
            return random.NextDouble() * 2 * epsilon - epsilon;
        }

        public static IEnumerable<Matrix<double>> RandomInitialiseWeights(int inputLayerSize, int outputLayerSize, 
            IEnumerable<int> hiddenLayersSizes)
        {
            var initialThetas = new List<Matrix<double>>();

            int previousLayerSize = inputLayerSize+1;
            int nextLayerSize = -1;
            Matrix<double> theta = null;

            // if there are hidden layers
            if(hiddenLayersSizes.Count() > 0)
            {
                nextLayerSize = hiddenLayersSizes.ElementAt(0);
                for (int i = 0; i < hiddenLayersSizes.Count(); i++)
                {
                    theta = GetNextTheta(nextLayerSize, previousLayerSize);
                    initialThetas.Add(theta);

                    previousLayerSize = hiddenLayersSizes.ElementAt(i) + 1;
                    if (i + 1 < hiddenLayersSizes.Count())
                    {
                        nextLayerSize = hiddenLayersSizes.ElementAt(i + 1);
                    }
                }
            }

            nextLayerSize = outputLayerSize;
            theta = GetNextTheta(nextLayerSize, previousLayerSize);
            initialThetas.Add(theta);


            return initialThetas;
        }

        private static Matrix<double> GetNextTheta(int nextLayerSize, int previousLayerSize)
        {
            double epsilon = Math.Sqrt(6) / Math.Sqrt(previousLayerSize + nextLayerSize);

            var sizeData = previousLayerSize * nextLayerSize;
            var data = new double[sizeData];
            for (int k = 0; k < sizeData; k++)
            {
                data[k] = GetRandom(previousLayerSize, nextLayerSize, epsilon);
            }

            var theta = new DenseMatrix(nextLayerSize, previousLayerSize, data);

            return theta;
        }

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

        public static Tuple<Matrix<double>, Matrix<double>> GradientDescent(CostFunctionWithThetaParameter func,
            Matrix<double> theta, double alpha, int numberIterations)
        {
            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();
            
            Matrix<double> JHistory = new DenseMatrix(numberIterations, 1);
            for (int i = 0; i < numberIterations; i++)
            {
                var res = func(theta);

                var h = res.Item1;
                var grad = res.Item2;
                JHistory[i, 0] = h;

                
                // "bold driver" - if we decrease the cost function, increase the learning rate by 5% but
                // in case when we increase the cost function, decrease the learning rate by 50%
                if (i > 0)
                {
                    if (JHistory[i, 0] < JHistory[i - 1, 0])
                    {
                        alpha += (double)0.05 * alpha;
                    }
                    else
                    {
                        alpha -= (double)0.5 * alpha;
                    }
                }

                theta = theta - grad * alpha;

                if (i > 0 && JHistory[i, 0] < JHistory[i - 1, 0] &&
                    Equalities.DoubleEquals(JHistory[i, 0], JHistory[i - 1, 0]))
                {
                    break;
                }  
            }
  
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

        public static Matrix<double> PackThetas(IEnumerable<Matrix<double>> thetas)
        {
            var allThetas = Enumerable.Empty<double>();

            foreach (var theta in thetas)
            {
                var data = theta.ToColumnWiseArray();
                allThetas = allThetas.Concat(data);
            }
            var resultTheta = new DenseMatrix(allThetas.Count(), 1, allThetas.ToArray());

            return resultTheta;
        }

        public static IEnumerable<Matrix<double>> UnpackThetas(Matrix<double> theta,
            int inputLayerSize, IEnumerable<int> hiddenLayersSizes, int outputLayerSize)
        {
            var thetas = new List<Matrix<double>>();

            Matrix<double> currentTheta = null;
            int previousLayerSize = inputLayerSize + 1;
            int nextLayerSize = -1;
            int startIndex = 0;
            int size = -1;

            if (hiddenLayersSizes.Count() > 0)
            {
                nextLayerSize = hiddenLayersSizes.ElementAt(0);
                size = nextLayerSize * previousLayerSize;

                for (int i = 0; i < hiddenLayersSizes.Count(); i++)
                {
                    currentTheta = MatriceSubRowMatrixAndReshape(theta,
                        nextLayerSize, previousLayerSize,
                        startIndex, nextLayerSize * previousLayerSize);
                    thetas.Add(currentTheta);

                    previousLayerSize = hiddenLayersSizes.ElementAt(i) + 1;
                    startIndex += size;

                    if (i + 1 < hiddenLayersSizes.Count())
                    {
                        nextLayerSize = hiddenLayersSizes.ElementAt(i + 1);
                        size = nextLayerSize * previousLayerSize;
                    }

                }
            }

            nextLayerSize = outputLayerSize;
            size = nextLayerSize * previousLayerSize;

            currentTheta = MatriceSubRowMatrixAndReshape(theta,
                        nextLayerSize, previousLayerSize,
                        startIndex, nextLayerSize * previousLayerSize);
            thetas.Add(currentTheta);

            return thetas;
        }

        public static ForwardPropagationResult ForwardPropagation(Matrix<double> X, 
            IEnumerable<Matrix<double>> thetas)
        {
            var a = new List<Matrix<double>>();
            var z = new List<Matrix<double>>();

            var ai = X;
            foreach (var theta in thetas)
            {
                var onesai = GetMatrixColumnOfOnes(ai.RowCount);
                
                var a1i = onesai.Append(ai);
                a.Add(a1i);

                var zi = (a1i * theta.Transpose());
                z.Add(zi);
                
                ai = Sigmoid(zi);
            }

            var h = ai; //the output of NN

            return new ForwardPropagationResult(h, a, z);
        }

        public static Tuple<double, Matrix<double>> BackPropagation(Matrix<double> X,
            Matrix<double> y, Matrix<double> theta,
            IEnumerable<int> hiddenLayersSizes, int outputLayerSize, double lambda)
        {
            int inputLayerSize = X.ColumnCount;
            int numberOfTestSamples = X.RowCount;
            
            var thetas = UnpackThetas(theta, inputLayerSize, 
                hiddenLayersSizes, outputLayerSize);
          
            /// feed forward
            var feedForwardResult = ForwardPropagation(X, thetas);

            var h = feedForwardResult.OutputActivation;
            var newY = GetOutputVectors(y, h.ColumnCount);

            /// back propagation
            double J = BackPropagationCostFunction(h, newY,
                inputLayerSize, hiddenLayersSizes, outputLayerSize,
                thetas, lambda, numberOfTestSamples);

            var gradient = GetBackPropagationGradient(feedForwardResult, newY,
                thetas, lambda, numberOfTestSamples);
           
            return Tuple.Create(J, gradient);
        }

        /// <summary>
        ///  
        /// Back Propagation Gradient for a NN with just one hidden layer
        /// (algorithm for the other hidden layers is the same as for Theta1Gradient)
        /// 
        ///   // back propagate from the last layer to last hidden layer
        ///     smalldelta3 = (h' - newY)';
        ///     t2 = lambda * Theta2/m;
        ///     t2(:,1) = 0;
        ///     Theta2Gradient = (smalldelta3' * a2) / m  + t2;
        ///
        ///   // back propagate from the last hidden layer to the previous layer (and so on)
        ///     smalldelta2 = (smalldelta3 * Theta2(:,2:end)) .* sigmoidGradient(z2);
        ///     t1 = lambda * Theta1/m;
        ///     t1(:,1) = 0;
        ///     Theta1Gradient = (smalldelta2' * a1) / m  + t1;
        ///
        /// </summary>
        private static Matrix<double> GetBackPropagationGradient(ForwardPropagationResult feedForward,
            Matrix<double> newY, IEnumerable<Matrix<double>> thetas, double lambda, int numberOfTestSamples)
        {
            var m = numberOfTestSamples;

            var thetasArrays = new List<double[]>();

            var h = feedForward.OutputActivation;
            var theta = thetas.Last();
            var delta = h - newY;

            var aCount = feedForward.NodesActivations.Count();
            
            var thetaArr = GetThetaGradient(theta, delta,
                feedForward.NodesActivations.ElementAt(aCount - 1), lambda, m);

            thetasArrays.Insert(0, thetaArr);

            for (int i = thetas.Count() - 2, j = 0; i >= 0; i--, j++)
            {
                var z = feedForward.NodesOutputs.ElementAt(j);
                var thetaWithoutFirstColumn = theta.SubMatrix(0, theta.RowCount, 1, theta.ColumnCount - 1);
                var r = delta * thetaWithoutFirstColumn;
                delta = r.PointwiseMultiply(SigmoidGradient(z));

                var thetaArray = GetThetaGradient(thetas.ElementAt(i), delta, 
                    feedForward.NodesActivations.ElementAt(i), lambda, m);

                thetasArrays.Insert(0, thetaArray);
            }

            var gradMatrix = new DenseMatrix(thetasArrays.Sum(t => t.Count()), 1, thetasArrays.SelectMany(t => t).ToArray());

            return gradMatrix;
        }

        private static double[] GetThetaGradient(Matrix<double> theta, 
            Matrix<double> delta, Matrix<double> ai, double lambda, int m)
        {
            var thetaGradient = delta.Transpose() * ai;

            var t = (lambda * theta) / m;
            var zerosForT = GetArrayOfNumber(0, t.RowCount);
            t.SetColumn(0, zerosForT);

            thetaGradient = (thetaGradient / m) + t;

            var thetaArray = thetaGradient.ToColumnWiseArray();
            return thetaArray;
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
            int inputLayerSize, IEnumerable<int> hiddenLayersSizes, int numberLabels,
            IEnumerable<Matrix<double>> thetas, double lambda, int numberOfTestSamples)
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

            double toAdd = GetRegularisationTerm(thetas, lambda, m);
            
            J = J + toAdd;

            return J;
        }

        private static double GetRegularisationTerm(IEnumerable<Matrix<double>> thetas,
            double lambda, double outputLayerSize)
        {
            var m = outputLayerSize;

            double thetaSum = 0;

            foreach (var theta in thetas)
            {
                for (int i = 0; i < theta.RowCount; i++)
                {
                    for (int j = 1; j < theta.ColumnCount; j++) // // without first column - bias term!
                    {
                        thetaSum += Math.Pow(theta[i, j], 2);
                    }
                }
            }

            var toAdd = ((double)(lambda / ((double)(2 * m)))) * (thetaSum);

            return toAdd;
        }

        public static Tuple<double, Matrix<double>> GetPredictions(Matrix<double> X, Matrix<double> y,
            IEnumerable<Matrix<double>> thetas)
        {
            var feedForward = NeuralNetwork.ForwardPropagation(X, thetas);
            var h = feedForward.OutputActivation;

            Matrix<double> predictions = new DenseMatrix(h.RowCount, 1);
            int countMatches = 0;
            for (int i = 0; i < h.RowCount; i++)
            {
                var max = h[i, 0];
                var predictedClass = 0;
                for (int j = 0; j < h.ColumnCount; j++)
                {
                    if (h[i, j] > max)
                    {
                        max = h[i, j];
                        predictedClass = j + 1;
                    }
                }
                if (predictedClass == y[i, 0])
                {
                    countMatches++;
                }
                predictions[i, 0] = predictedClass;
            }

            var percent = (countMatches / (double)y.RowCount) * 100;

            return Tuple.Create(percent, predictions);
        }

        private static void Populate<T>(this T[] arr, T value)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                arr[i] = value;
            }
        }

        private static Matrix<double> GetOutputVectors(Matrix<double> yClasses, int numberLabels)
        {
            int m = yClasses.RowCount;

            var y = new DenseMatrix(m, numberLabels);

            for (int i = 0; i < m; i++)
            {
                y[i, (int)yClasses[i, 0] - 1] = 1;
            }

            return y;
        }
    }
}
