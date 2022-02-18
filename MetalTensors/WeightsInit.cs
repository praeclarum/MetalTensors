using System;
using System.Threading.Tasks;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors
{
    public abstract class WeightsInit
    {
        public static WeightsInit Default = new GlorotUniformInit (1.0f);

        public static WeightsInit GlorotUniform (float scale = 1.0f) => new GlorotUniformInit (scale);
        public static WeightsInit Normal (float mean, float standardDeviation) => new NormalInit (mean, standardDeviation);
        public static WeightsInit Uniform (float min, float max) => new UniformInit (min, max);

        public abstract Task InitWeightsAsync (MPSVector vector, int seed, int fanIn, int fanOut);

        class GlorotUniformInit : WeightsInit
        {
            // https://github.com/keras-team/keras/blob/8e76f053e8823988626e74cb386fc01ec857859a/keras/initializers/initializers_v2.py#L503
            // https://github.com/keras-team/keras/blob/8e76f053e8823988626e74cb386fc01ec857859a/keras/initializers/initializers_v2.py#L687

            readonly float weightScale;

            public GlorotUniformInit (float weightScale)
            {
                this.weightScale = weightScale;
            }

            public override Task InitWeightsAsync (MPSVector vector, int seed, int fanIn, int fanOut)
            {
                var scale = weightScale / Math.Max (1.0, (fanIn + fanOut) / 2.0);
                var limit = Math.Sqrt (3.0 * scale);
                //Console.WriteLine ($"LIMIT {limit}");
                return vector.UniformInitAsync ((float)-limit, (float)limit, seed, downloadToCpu: true);
            }
        }

        class NormalInit : WeightsInit
        {
            readonly float mean;
            readonly float standardDeviation;

            public NormalInit (float mean, float standardDeviation)
            {
                this.mean = mean;
                this.standardDeviation = standardDeviation;
            }

            public override Task InitWeightsAsync (MPSVector vector, int seed, int fanIn, int fanOut)
            {
                return vector.NormalInitAsync (mean, standardDeviation, seed, downloadToCpu: true);
            }
        }

        class UniformInit : WeightsInit
        {
            readonly float min;
            readonly float max;

            public UniformInit (float min, float max)
            {
                this.min = min;
                this.max = max;
            }

            public override Task InitWeightsAsync (MPSVector vector, int seed, int fanIn, int fanOut)
            {
                return vector.UniformInitAsync (min, max, seed, downloadToCpu: true);
            }
        }
    }    
}
