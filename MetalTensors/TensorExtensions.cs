using System;
using System.Collections.Generic;

namespace MetalTensors
{
    public static class TensorExtensions
    {
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

        public static Tensor[] Map (this Tensor[] tensors, Dictionary<Tensor, Tensor> map)
        {
            var n = tensors.Length;
            if (n == 0)
                return Array.Empty<Tensor> ();

            var r = new Tensor[n];
            for (var i = 0; i < n; i++) {
                if (map.TryGetValue (tensors[i], out var nt)) {
                    r[i] = nt;
                }
                else {
                    r[i] = tensors[i].MapInputs (map);
                }
            }
            return r;
        }
    }
}
