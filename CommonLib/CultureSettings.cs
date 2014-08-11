using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CommonLib
{
    public static class CultureSettings
    {
        public static void SetNumberDecimalSeparatorToDot()
        {
            var customCulture = (System.Globalization.CultureInfo)System.Threading.Thread.CurrentThread.CurrentCulture.Clone();
            customCulture.NumberFormat.NumberDecimalSeparator = ".";

            System.Threading.Thread.CurrentThread.CurrentCulture = customCulture;
        }
    }
}
