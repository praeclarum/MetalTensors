using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using Foundation;
using Metal;
using MetalPerformanceShaders;

using static MetalTensors.MetalHelpers;

namespace MetalTensors.Layers
{
    public interface IWeightsDataSource
    {
        void SetOptimizationOptions (bool trainable, float learningRate);
    }

    public abstract class WeightsLayer : TrainableLayer
    {
        public Weights Weights { get; }

        static readonly ConcurrentDictionary<IntPtr, IMTLCommandQueue> deviceQueues =
            new ConcurrentDictionary<IntPtr, IMTLCommandQueue> ();

        readonly ConcurrentDictionary<IntPtr, IWeightsDataSource> deviceDataSources =
            new ConcurrentDictionary<IntPtr, IWeightsDataSource> ();

        public WeightsLayer (string? name = null, bool isTrainable = true, Weights? weights = null)
            : base (name, isTrainable: isTrainable)
        {
            Weights = weights ?? new Weights ();
        }

        public override Config Config => base.Config.Update (new Config {
            { "weights", Weights },
        });

        public IWeightsDataSource? TryGetDataSource (IMTLDevice device)
        {
            var key = device.Handle;
            if (deviceDataSources.TryGetValue (key, out var w))
                return w;
            return null;
        }

        public T GetDataSource<T> (IMTLDevice device) where T : IWeightsDataSource
        {
            var key = device.Handle;
            if (deviceDataSources.TryGetValue (key, out var w))
                return (T)w;

            if (!deviceQueues.TryGetValue (key, out var queue)) {
                queue = device.CreateCommandQueue ();
                if (queue == null)
                    throw new Exception ($"Failed to create queue to load values");
                if (!deviceQueues.TryAdd (key, queue)) {
                    queue = deviceQueues[key];
                }
            }
            w = CreateDataSource (queue);
            if (deviceDataSources.TryAdd (key, w))
                return (T)w;
            return (T)deviceDataSources[key];
        }

        protected abstract IWeightsDataSource CreateDataSource (IMTLCommandQueue device);        
    }
}
