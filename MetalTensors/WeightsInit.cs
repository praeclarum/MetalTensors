using System;
using System.Threading.Tasks;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors
{
    public abstract class WeightsInit : Configurable
    {
        public static WeightsInit Default = new GlorotUniformInit (scale: 1.0f);

        public static WeightsInit GlorotUniform (float scale = 1.0f) => new GlorotUniformInit (scale);
        public static WeightsInit Normal (float mean, float standardDeviation) => new NormalInit (mean, standardDeviation);
        public static WeightsInit Uniform (float min, float max) => new UniformInit (min, max);

        public abstract Task InitWeightsAsync (MPSVector vector, int seed, int fanIn, int fanOut, IMTLCommandQueue queue);
    }

    public class GlorotUniformInit : WeightsInit
    {
        // https://github.com/keras-team/keras/blob/8e76f053e8823988626e74cb386fc01ec857859a/keras/initializers/initializers_v2.py#L503
        // https://github.com/keras-team/keras/blob/8e76f053e8823988626e74cb386fc01ec857859a/keras/initializers/initializers_v2.py#L687

        public float Scale { get; }

        public GlorotUniformInit (float scale = 1.0f)
        {
            Scale = scale;
        }

        public override Config Config => base.Config.Add ("scale", 1.0f);

        public override Task InitWeightsAsync (MPSVector vector, int seed, int fanIn, int fanOut, IMTLCommandQueue queue)
        {
            var scale = Scale / Math.Max (1.0, (fanIn + fanOut) / 2.0);
            var limit = Math.Sqrt (3.0 * scale);
            //Console.WriteLine ($"LIMIT {limit}");
            return vector.UniformInitAsync ((float)-limit, (float)limit, seed, downloadToCpu: true, queue: queue);
        }
    }

    public class NormalInit : WeightsInit
    {
        public float Mean { get; }
        public float StandardDeviation { get; }

        public NormalInit (float mean, float standardDeviation)
        {
            Mean = mean;
            StandardDeviation = standardDeviation;
        }

        public override Config Config => base.Config.Update (new Config {
            { "mean", Mean },
            { "standardDeviation", StandardDeviation },
        });

        public override Task InitWeightsAsync (MPSVector vector, int seed, int fanIn, int fanOut, IMTLCommandQueue queue)
        {
            return vector.NormalInitAsync (Mean, StandardDeviation, seed, downloadToCpu: true, queue: queue);
        }
    }

    public class UniformInit : WeightsInit
    {
        public float Min { get; }
        public float Max { get; }

        public UniformInit (float min, float max)
        {
            Min = min;
            Max = max;
        }

        public override Config Config => base.Config.Update (new Config {
            { "min", Min },
            { "max", Max },
        });

        public override Task InitWeightsAsync (MPSVector vector, int seed, int fanIn, int fanOut, IMTLCommandQueue queue)
        {
            return vector.UniformInitAsync (Min, Max, seed, downloadToCpu: true, queue: queue);
        }
    }
}
