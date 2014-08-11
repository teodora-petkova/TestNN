using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace CommonLib
{
    public class Matrix
    {
        #region members

        public int Columns = 0;
        public int Rows = 0;

        private double[,] data;

        #endregion

        #region constructors

        public Matrix()
        {
            this.data = null;
        }

        public Matrix(int rows, int cols)
        {
            Columns = cols;
            Rows = rows;

            this.data = new double[rows, cols];
            this.Zero();
        }

        public Matrix(int numberToInitialise, int rows, int cols)
        {
            Columns = cols;
            Rows = rows;

            this.data = new double[rows, cols];
            this.Initialise(numberToInitialise);
        }

        public Matrix(int rows, int cols, double[] inputData)
        {
            Columns = cols;
            Rows = rows;

            int size = cols * rows;

            this.data = new double[rows, cols];

            if (inputData.Count() != size)
            {
                throw new Exception("Size of the matrix array is not matching!");
            }

            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    this.data[i, j] = inputData[i * Columns + j];
                }
            }
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        public Matrix(Matrix m)
        {
            Set(m);
        }

        /// <summary>
        /// Set this matric to m.
        /// </summary>
        private void Set(Matrix m)
        {
            Resize(m.Columns, m.Rows);
            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = 0; j < m.Columns; j++)
                {
                    this.data[i, j] = m[i, j];
                }
            }
        }

        #endregion

        // index operator
        public double this[int x, int y]
        {
            get
            {
                return this.data[x, y];
            }
            set
            {
                this.data[x, y] = value;
            }
        }

        /// <summary>
        /// Resize.
        /// </summary>
        public void Resize(int cols, int rows)
        {
            if ((Columns == cols) && (Rows == rows)) return;

            Columns = cols;
            Rows = rows;

            this.data = new double[rows, cols];
            this.Zero();
        }

        /// <summary>
        /// Clone this matrix.
        /// </summary>
        /// <returns></returns>
        public Matrix Clone()
        {
            Matrix m = new Matrix();
            m.Set(this);
            return m;
        }

        /// <summary>
        /// In place scalar multiplication.
        /// this *= scalar.
        /// </summary>
        public void Multiply(double scalar)
        {
            for (int i = 0; i < this.Rows; i++)
            {
                for (int j = 0; j < this.Columns; j++)
                {
                    this.data[i, j] *= scalar;
                }
            }
        }

        /// <summary>
        /// Scalar multiplication.
        /// result = Multiply(matrix, scalar);
        /// </summary>
        public static Matrix Multiply(Matrix m, double scalar)
        {
            Matrix rv = m.Clone();
            rv.Multiply(scalar);
            return rv;
        }

        /// <summary>
        /// Scalar multiplication.
        /// result = Multiply(matrix, scalar);
        /// </summary>
        public static Matrix operator *(Matrix m, double scalar)
        {
            return Multiply(m, scalar);
        }

        /// <summary>
        /// Multiply two matrixes.
        /// </summary>
        public static Matrix Multiply(Matrix a, Matrix b)
        {
            Matrix result = null;

            if (a != null && b != null && a.Columns == b.Rows)
            {
                result = new Matrix(a.Rows, b.Columns);

                for (int i = 0; i < a.Rows; i++)
                {
                    for (int j = 0; j < b.Columns; j++)
                    {
                        double s = 0;
                        for (int k = 0; k < a.Columns; k++)
                        {
                            double aVar = a[i, k];
                            double bVar = b[k, j];
                            s += aVar * bVar;
                        }
                        result[i, j] = s;
                    }
                }
            }

            return result;
        }

        public static Matrix operator *(Matrix a, Matrix b)
        {
            return Multiply(a, b);
        }

        /// <summary>
        /// Multiply in place this * b.
        /// </summary>
        public void Multiply(Matrix b)
        {
            Matrix tmp = Matrix.Multiply(this, b);
            this.Set(tmp);
        }

        /// <summary>
        /// Result = a*b*a^T.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Matrix MultiplyABAT(Matrix a, Matrix b)
        {
            Matrix result = Multiply(a, b);
            Matrix aTransposed = Matrix.Transpose(a);
            result.Multiply(aTransposed);
            return result;
        }

        /// <summary>
        /// Add scalar.
        /// </summary>
        public static Matrix Add(Matrix a, double scalar)
        {
            Matrix result = new Matrix(a);
            result.Add(scalar);
            return result;
        }

        public static Matrix operator +(Matrix a, double scalar)
        {
            return Add(a, scalar);
        }

        /// <summary>
        /// Add scalar in place
        /// </summary>
        public void Add(double scalar)
        {
            for (int i = 0; i < this.Rows; i++)
            {
                for (int j = 0; j < this.Columns; j++)
                {
                    this.data[i, j] += scalar;
                }
            }
        }

        /// <summary>
        /// Add matrix.
        /// </summary>
        public static Matrix Add(Matrix a, Matrix b)
        {
            Matrix result = new Matrix(a);
            result.Add(b);
            return result;
        }

        public static Matrix operator +(Matrix a, Matrix b)
        {
            return Add(a, b);
        }

        /// <summary>
        /// Add matrix in place
        /// </summary>
        public void Add(Matrix a)
        {
            for (int i = 0; i < this.Rows; i++)
            {
                for (int j = 0; j < this.Columns; j++)
                {
                    this.data[i, j] += a[i, j];
                }
            }
        }

        /// <summary>
        /// Subtract scalar.
        /// </summary>
        public static Matrix Subtract(Matrix a, double scalar)
        {
            Matrix result = new Matrix(a);
            result.Subtract(scalar);
            return result;
        }

        public static Matrix operator -(Matrix a, double scalar)
        {
            return Subtract(a, scalar);
        }

        /// <summary>
        /// Subtract scalar in place
        /// </summary>
        public void Subtract(double scalar)
        {
            for (int i = 0; i < this.Rows; i++)
            {
                for (int j = 0; j < this.Columns; j++)
                {
                    this.data[i, j] -= scalar;
                }
            }
        }

        /// <summary>
        /// Subtract matrix b from a.
        /// </summary>
        public static Matrix Subtract(Matrix a, Matrix b)
        {
            Matrix result = new Matrix(a);
            result.Subtract(b);
            return result;
        }

        public static Matrix operator -(Matrix a, Matrix b)
        {
            return Subtract(a, b);
        }
        /// <summary>
        /// Subtract matrix in place
        /// </summary>
        public void Subtract(Matrix a)
        {
            for (int i = 0; i < this.Rows; i++)
            {
                for (int j = 0; j < this.Columns; j++)
                {
                    this.data[i, j] -= a[i, j];
                }
            }
        }

        /// <summary>
        /// Transpose matrix m.
        /// </summary>
        public static Matrix Transpose(Matrix m)
        {
            Matrix result = new Matrix(m.Columns, m.Rows);
            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = 0; j < m.Columns; j++)
                {
                    result[j, i] = m[i, j];
                }
            }
            return result;
        }

        /// <summary>
        /// Transpose this matrix in place.
        /// </summary>
        public void Transpose()
        {
            Matrix result = Transpose(this);
            this.Set(result);
        }

        public static Matrix operator ~(Matrix a)
        {
            return Transpose(a);
        }

        /// <summary>
        /// Test if this is an identity matrix.
        /// </summary>
        /// <returns></returns>
        public bool IsIdentity()
        {
            if (Columns != Rows)
            {
                return false;
            }

            for (int i = 0; i < this.Rows; i++)
            {
                for (int j = 0; j < this.Columns; j++)
                {
                    if (i == j && this.data[i, j] != 1)
                    {
                        return false;
                    }

                    if (i != j && this.data[i, j] != 0)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        /// <summary>
        /// Set an identity matrix.
        /// </summary>
        public void SetIdentity()
        {
            if (Columns != Rows)
            {
                return;
            }

            for (int i = 0; i < this.Rows; i++)
            {
                for (int j = 0; j < this.Columns; j++)
                {
                    this.data[i, j] = (j == i) ? 1 : 0;
                }
            }
        }

        /// <summary>
        /// Zero.
        /// </summary>
        public void Zero()
        {
            for (int i = 0; i < this.Rows; i++)
            {
                for (int j = 0; j < this.Columns; j++)
                {
                    this.data[i, j] = 0;
                }
            }
        }

        public void Zero(int num, bool isRow = true)
        {
            if (isRow)
            {
                for (int j = 0; j < this.Columns; j++)
                {
                    this.data[num, j] = 0;
                }
            }
            else
            {
                for (int i = 0; i < this.Rows; i++)
                {
                    this.data[i, num] = 0;
                }
            }
        }

        /// <summary>
        /// Ones.
        /// </summary>
        public void Initialise(int numberToInitialise)
        {
            for (int i = 0; i < this.Rows; i++)
            {
                for (int j = 0; j < this.Columns; j++)
                {
                    this.data[i, j] = numberToInitialise;
                }
            }
        }

        /// <summary>
        /// Determinant.
        /// </summary>
        public double Determinant
        {
            get
            {
                if (Columns != Rows) return 0;

                if (Columns == 0) return 0;
                if (Columns == 1) return this.data[0, 0];
                if (Columns == 2) return (this.data[0, 0] * this.data[1, 1]) - (this.data[0, 1] * this.data[1, 0]);
                if (Columns == 3) return
                    (this.data[0, 0] * ((this.data[1, 1] * this.data[2, 2]) - (this.data[1, 2] * this.data[2, 1]))) -
                    (this.data[0, 1] * ((this.data[1, 0] * this.data[2, 2]) - (this.data[1, 2] * this.data[2, 0]))) +
                    (this.data[0, 2] * ((this.data[1, 0] * this.data[2, 1]) - (this.data[1, 1] * this.data[2, 0])));

                // only supporting 1x1, 2x2 and 3x3
                return 0;
            }
        }

        /// <summary>
        /// Invert. (only supporting 1x1, 2x2 and 3x3)
        /// </summary>
        public static Matrix Invert(Matrix m)
        {
            if (m.Columns != m.Rows) return null;
            double det = m.Determinant;
            if (det == 0) return m;

            Matrix result = new Matrix(m);
            if (m.Columns == 1)
            {
                result[0, 0] = 1 / result[0, 0];
            }
            det = 1 / det;
            if (m.Columns == 2)
            {
                result[0, 0] = det * m[1, 1];
                result[0, 1] = -det * m[1, 0];
                result[1, 1] = det * m[0, 0];
                result[1, 0] = -det * m[0, 1];
            }
            if (m.Columns == 3)
            {
                m.Transpose(); // transpose the matrix!

                result[0, 0] = det * ((m[1, 1] * m[2, 2]) - (m[1, 2] * m[2, 1]));
                result[0, 1] = -det * ((m[1, 0] * m[2, 2]) - (m[1, 2] * m[2, 0]));
                result[0, 2] = det * ((m[1, 0] * m[2, 1]) - (m[1, 1] * m[2, 0]));

                result[1, 0] = -det * ((m[0, 1] * m[2, 2]) - (m[0, 2] * m[2, 1]));
                result[1, 1] = det * ((m[0, 0] * m[2, 2]) - (m[0, 2] * m[2, 0]));
                result[1, 2] = -det * ((m[0, 0] * m[2, 1]) - (m[0, 1] * m[2, 0]));

                result[2, 0] = det * ((m[0, 1] * m[1, 2]) - (m[0, 2] * m[1, 1]));
                result[2, 1] = -det * ((m[0, 0] * m[1, 2]) - (m[0, 2] * m[1, 0]));
                result[2, 2] = det * ((m[0, 0] * m[1, 1]) - (m[0, 1] * m[1, 0]));
            }
            return result;
        }

        public static Matrix operator !(Matrix a)
        {
            return Invert(a);
        }

        public static Matrix operator ^(double p, Matrix a)
        {
            var b = new Matrix(a.Rows, a.Columns);

            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                {
                    b[i, j] = Math.Pow(p, a[i, j]);
                }
            }

            return b;
        }

        public static Matrix operator ^(Matrix a, double p)
        {
            var b = new Matrix(a.Rows, a.Columns);

            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                {
                    b[i, j] = Math.Pow(a[i, j], p);
                }
            }

            return b;
        }

        public static Matrix operator /(double p, Matrix a)
        {
            var b = new Matrix(a.Rows, a.Columns);

            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                {
                    b[i, j] = p / a[i, j];
                }
            }

            return b;
        }

        public static Matrix operator /(Matrix a, double p)
        {
            var b = new Matrix(a.Rows, a.Columns);

            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                {
                    b[i, j] = (double)(a[i, j] / (double)p);
                }
            }

            return b;
        }

        public static Matrix Log(Matrix a)
        {
            var b = new Matrix(a.Rows, a.Columns);

            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                {
                    b[i, j] = Math.Log(a[i, j]);
                }
            }

            return b;
        }

        public static Matrix Sum(Matrix a, bool isColumns = true) // false => isRows
        {
            Matrix b = new Matrix();

            if (isColumns)
            {
                b = new Matrix(1, a.Columns);

                for (int j = 0; j < a.Columns; j++)
                {
                    for (int i = 0; i < a.Rows; i++)
                    {
                        b[0, j] += a[i, j];
                    }
                }
            }
            else
            {
                b = new Matrix(a.Rows, 1);

                for (int i = 0; i < a.Rows; i++)
                {
                    for (int j = 0; j < a.Columns; j++)
                    {
                        b[i, 0] += a[i, j];
                    }
                }
            }

            return b;
        }

        public void Save(string fileName)
        {
            using (var file = new System.IO.StreamWriter(fileName))
            {
                for (int i = 0; i < this.Rows; i++)
                {
                    for (int j = 0; j < this.Columns; j++)
                    {
                        if (j != 0)
                        {
                            file.Write(" ");
                        }
                        file.Write(this[i, j]);
                    }
                    file.WriteLine();
                }
            }
        }

        public Matrix SubMatrix(int start, int end, bool isColumns = true)
        {
            Matrix b = new Matrix();

            if (isColumns)
            {
                b = new Matrix(this.Rows, end - start);

                for (int j = 0; j + start < end; j++)
                {
                    for (int i = 0; i < this.Rows; i++)
                    {
                        b[i, j] = this[i, j + start];
                    }
                }
            }
            else
            {
                b = new Matrix(end - start, this.Columns);

                for (int i = 0; i + start < end; i++)
                {
                    for (int j = 0; j < this.Columns; j++)
                    {
                        b[i, j] = this[i + start, j];
                    }
                }
            }

            return b;
        }

        public static Matrix ConcatMatrices(Matrix a, Matrix b, bool isColumns = true)
        {
            Matrix c = null;

            if (isColumns)
            {
                if (a.Rows != b.Rows)
                {
                    return c;
                }
                else
                {
                    c = new Matrix(a.Rows, a.Columns + b.Columns);

                    int column = 0;
                    for (column = 0; column < a.Columns; column++)
                    {
                        for (int i = 0; i < a.Rows; i++)
                        {
                            c[i, column] = a[i, column];
                        }
                    }
                    for (int j = column; j < column + b.Columns; j++)
                    {
                        for (int i = 0; i < b.Rows; i++)
                        {
                            c[i, j] = b[i, j - column];
                        }
                    }
                }
            }
            else
            {
                if (a.Columns != b.Columns)
                {
                    return c;
                }
                else
                {
                    c = new Matrix(a.Rows + b.Rows, a.Columns);

                    int row = 0;
                    for (row = 0; row < a.Rows; row++)
                    {
                        for (int j = 0; j < a.Columns; j++)
                        {
                            c[row, j] = a[row, j];
                        }
                    }
                    for (int i = row; i < row + b.Rows; i++)
                    {
                        for (int j = 0; j < b.Columns; j++)
                        {
                            c[i, j] = b[i - row, j];
                        }
                    }
                }
            }

            return c;
        }

        public static Matrix LoadMatrixFromFile(string filePath)
        {
            Matrix m = new Matrix();

            using (StreamReader sr = new StreamReader(filePath))
            {
                String line;
                
                var matrix = new List<List<double>>();
                var row = new List<double>();

                while ((line = sr.ReadLine()) != null)
                {
                    var nums = line.Split(' ');
                    row = new List<double>();

                    foreach (var num in nums)
                    { 
                        double number;
                        if(Double.TryParse(num, out number))
                        {
                            row.Add(number);
                        }
                    }
                    matrix.Add(row);
                }

                m = new Matrix(matrix.Count, row.Count);
                for(int i = 0; i < m.Rows; i++)
                { 
                    for(int j = 0; j < m.Columns; j++)
                    {
                        m[i, j] = matrix[i].ElementAt(j);
                    }
                }
            }
            return m;
        }

        public bool Equals(Matrix m, double difference = 0.00000000001)
        {
            bool result = true;
            
            for (int i = 0; i < this.Rows; i++)
            {
                for (int j = 0; j < this.Columns; j++)
                {

                    if (Math.Abs(this.data[i, j] - m[i, j]) <= difference)
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

        public static Matrix operator -(double p, Matrix m)
        {
            return (m - p)*(-1);
        }

        public static Matrix operator *(double p, Matrix m)
        {
            return m * p;
        }

        public void Reshape(int numRows, int numColumns)
        {
          
            if (this.Rows * this.Columns != numRows * numColumns)
            {
                throw new Exception("Incorrect demension for reshape!");
            }
            else
            {
                var array1d = this.GetSingleArrayAsColumns();
                var m = new Matrix(numRows, numColumns);
                for (int j = 0; j < numColumns; j++)
                {
                    for (int i = 0; i < numRows; i++)      
                    {
                        m[i, j] = array1d[i+ j*numRows];
                    }
                }

                this.Set(m);
            }
        }

        public double[] GetSingleArrayAsRows()
        {
            var array2d = this.data;

            var array1d = new double[array2d.GetLength(0) * array2d.GetLength(1)];
            var current = 0;
            for (int i = 0; i < array2d.GetLength(0); i++)
            {
                for (int j = 0; j < array2d.GetLength(1); j++)
                {
                    array1d[current++] = array2d[i, j];
                }
            }
            return array1d;
        }

        public double[] GetSingleArrayAsColumns()
        {
            var array2d = this.data;

            var array1d = new double[array2d.GetLength(0) * array2d.GetLength(1)];
            var current = 0;
            for (int j = 0; j < array2d.GetLength(1); j++)
            {
                for (int i = 0; i < array2d.GetLength(0); i++)
                {
                    array1d[current++] = array2d[i, j];
                }
            }
            return array1d;
        }

        public Matrix MultiplyElementWise(Matrix b)
        {
            Matrix a = this;

            if (a.Columns != b.Columns || a.Rows != b.Rows)
            {
                throw new Exception("Not supported matrix dimensions!");
            }

            var m = new Matrix(a.Rows, a.Columns);
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                {
                    m[i, j] = a[i, j] * b[i, j];
                }
            }
            return m;
        }
    }
}
