using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using Foundation;
using Metal;
using MetalPerformanceShaders;

using static MetalTensors.MetalHelpers;

namespace MetalTensors.Layers
{
    public interface IWeightsDataSource { }

    public abstract class WeightsLayer : TrainableLayer
    {
        public Weights Weights { get; } = new Weights ();

        readonly ConcurrentDictionary<IntPtr, IWeightsDataSource> deviceDataSources =
            new ConcurrentDictionary<IntPtr, IWeightsDataSource> ();

        public WeightsLayer (string? name = null, bool isTrainable = true)
            : base (name, isTrainable)
        {

        }

        public override Config Config => base.Config.Update (new Config {
            { "weights", Weights },
        });

        public T GetDataSource<T> (IMTLDevice device) where T : IWeightsDataSource
        {
            var key = device.Handle;
            if (deviceDataSources.TryGetValue (key, out var w))
                return (T)w;
            w = CreateDataSource (device);
            if (deviceDataSources.TryAdd (key, w))
                return (T)w;
            return (T)deviceDataSources[key];
        }

        protected abstract IWeightsDataSource CreateDataSource (IMTLDevice device);
    }
}
