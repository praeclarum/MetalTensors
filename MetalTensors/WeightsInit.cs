using System;
using System.Threading.Tasks;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors
{
    public abstract class WeightsInit
    {
        //public static WeightsInit Default = new UniformInit (-0.2f, 0.2f);
        public static WeightsInit Default = new GaussianInit (0, 0.1f);

        public static WeightsInit Uniform (float min, float max) => new UniformInit (min, max);
        public static WeightsInit Gaussian (float mean, float standardDeviation) => new GaussianInit (mean, standardDeviation);

        public abstract float[] GetWeights (int seed, int length);

        public abstract Task InitWeightsAsync (MPSVector vector, int seed);

        class UniformInit : WeightsInit
        {
            readonly float min;
            readonly float max;

            public UniformInit (float min, float max)
            {
                this.min = min;
                this.max = max;
            }

            public override float[] GetWeights (int seed, int length)
            {
                var rand = new Random (seed);
                var r = new float[length];

                var d = max - min;

                for (var i = 0; i < length; i++) {
                    double u = rand.NextDouble ();
                    double randUniform = min + d * u;
                    r[i] = (float)randUniform;
                }

                return r;
            }

            public override Task InitWeightsAsync (MPSVector vector, int seed)
            {
                return vector.UniformInitAsync (min, max, seed, downloadToCpu: true);
            }
        }

        class GaussianInit : WeightsInit
        {
            readonly float mean;
            readonly float standardDeviation;

            public GaussianInit (float mean, float standardDeviation)
            {
                this.mean = mean;
                this.standardDeviation = standardDeviation;
            }

            public override float[] GetWeights (int seed, int length)
            {
                var rand = new Random (seed);
                var r = new float[length];

                for (var i = 0; i < length; i++) {
                    double u1 = 1.0 - rand.NextDouble ();
                    double u2 = 1.0 - rand.NextDouble ();
                    double randStdNormal = Math.Sqrt (-2.0 * Math.Log (u1)) * Math.Sin (2.0 * Math.PI * u2);
                    double randNormal = mean + standardDeviation * randStdNormal;
                    r[i] = (float)randNormal;
                }

                return r;
            }

            public override Task InitWeightsAsync (MPSVector vector, int seed)
            {
                return vector.NormalInitAsync (mean, standardDeviation, seed, downloadToCpu: true);
            }
        }
    }    
}
