using System;
using System.Collections.Generic;

namespace MetalTensors
{
    public static class Shapes
    {
        const int ncache = 1024;

        static readonly int[]?[] shapes1 = new int[ncache][];
        public static int[] GetShape (nuint dim0) => GetShape ((int)dim0);
        public static int[] GetShape (int dim0)
        {
            if (dim0 < shapes1.Length) {
                var s = shapes1[dim0];
                if (s != null)
                    return s;
                s = new int[1] { dim0 };
                shapes1[dim0] = s;
                return s;
            }
            else {
                return new int[1] { dim0 };
            }
        }

        static readonly int[]?[]?[]?[] shapes3 = new int[ncache][][][];
        public static int[] GetShape (nuint dim0, nuint dim1, nuint dim2) => GetShape ((int)dim0, (int)dim1, (int)dim2);
        public static int[] GetShape (int dim0, int dim1, int dim2)
        {
            if (dim0 < ncache && dim1 < ncache && dim2 < ncache) {
                var s1 = shapes3[dim0];
                if (s1 == null) {
                    s1 = new int[ncache][][];
                    shapes3[dim0] = s1;
                }
                var s2 = s1[dim1];
                if (s2 == null) {
                    s2 = new int[ncache][];
                    s1[dim1] = s2;
                }
                var s3 = s2[dim2];
                if (s3 == null) {
                    s3 = new int[3] { dim0, dim1, dim2 };
                    s2[dim2] = s3;
                }
                return s3;
            }
            else {
                return new int[1] { dim0 };
            }
        }
    }

    public static class ShapeExtensions
    {
        public static string ToShapeString (this int[] shape)
        {
            return "(" + string.Join (", ", shape) + ")";
        }

        public static int GetShapeLength (this int[] shape)
        {
            var r = 1;
            for (var i = 0; i < shape.Length; i++) {
                r *= shape[i];
            }
            return r;
        }

        static readonly int[] oneShape = { 1 };

        public static int[] NormalizeShape (this int[]? shape)
        {
            if (shape is null)
                return oneShape;

            var n = shape.Length;
            if (n <= 0)
                return oneShape;

            if (n == 1)
                return Shapes.GetShape (Math.Max (1, shape[0]));

            for (var bi = 0; bi < n; bi++) {
                if (shape[bi] <= 0) {
                    var ns = new int[n];
                    for (var i = 0; i < n; i++) {
                        ns[i] = Math.Max (1, shape[i]);
                    }
                    return ns;
                }
            }
            return shape;
        }

        public static bool ShapeEquals (this int[] shape, int[] other)
        {
            if (shape.Length != other.Length)
                return false;
            for (var i = 0; i < shape.Length; i++) {
                if (shape[i] != other[i])
                    return false;
            }
            return true;
        }
    }
}
