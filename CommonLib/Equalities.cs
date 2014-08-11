using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CommonLib
{
    public class Equalities
    {
        public static bool DoubleEquals(double a, double b, double difference = 0.0000000000001)
        {
            bool result = true;

            if (Math.Abs(a - b) <= difference)
            {
                result &= true;
            }
            else
            {
                result &= false;
            }

            return result;
        }
    }
}
