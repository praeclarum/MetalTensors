using System;
using System.Collections.Generic;

namespace MetalTensors
{
    public static class TensorExtensions
    {
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
