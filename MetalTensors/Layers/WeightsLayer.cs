using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading.Tasks;
using Foundation;
using Metal;
using MetalPerformanceShaders;

using static MetalTensors.MetalHelpers;

namespace MetalTensors.Layers
{
    public interface IWeightsDataSource
    {
        void SetOptimizationOptions (bool trainable, float learningRate);
        bool DownloadWeightsFromGpu ();
    }

    public abstract class WeightsLayer : TrainableLayer, IHasBuffers
    {
        static readonly ConcurrentDictionary<IntPtr, IMTLCommandQueue> deviceQueues =
            new ConcurrentDictionary<IntPtr, IMTLCommandQueue> ();

        readonly ConcurrentDictionary<IntPtr, IWeightsDataSource> deviceDataSources =
            new ConcurrentDictionary<IntPtr, IWeightsDataSource> ();

        /// <summary>
        /// Values stored after deserialization or a memory purge.
        /// </summary>
        readonly ConcurrentDictionary<string, float[]?> weightValues = new ConcurrentDictionary<string, float[]?> ();

        /// <summary>
        /// Values represented by Metal. Data is reflected between the CPU and GPU with manual synchronization.
        /// </summary>
        readonly ConcurrentDictionary<string, MPSVector> weightVectors = new ConcurrentDictionary<string, MPSVector> ();

        public string[] ParameterNames { get; }

        protected WeightsLayer (string? name, bool isTrainable, string[] parameterNames)
            : base (name, isTrainable: isTrainable)
        {
            ParameterNames = parameterNames;
        }

        public void AddParameter (string parameterName, MPSVector vector, float initialValue)
        {
            if (weightValues.TryGetValue (parameterName, out var memory) && memory != null) {
                vector.Init (memory);
            }
            else {
                vector.Fill (initialValue);
                weightVectors[parameterName] = vector;
            }
        }

        public void AddParameter (string parameterName, OptimizableVector vectors, float initialValue)
        {
            var vector = vectors.Value;
            if (weightValues.TryGetValue (parameterName, out var memory) && memory != null) {
                vector.Init (memory);
            }
            else {
                vector.Fill (initialValue);
                weightVectors[parameterName] = vector;
            }
        }

        public async Task AddParameterAsync (string parameterName, OptimizableVector vectors, WeightsInit initialValue, int fanIn, int fanOut, IMTLCommandQueue queue)
        {
            var vector = vectors.Value;
            if (weightValues.TryGetValue (parameterName, out var memory) && memory != null) {
                vector.Init (memory);
            }
            else {
                var seed = (int)DateTime.Now.Ticks;
                await initialValue.InitWeightsAsync (vector, seed, fanIn: fanIn, fanOut: fanOut, queue: queue).ConfigureAwait (false);
                weightVectors[parameterName] = vector;
            }
        }

        public void ReadBuffers (ReadBuffer reader)
        {
            foreach (var p in ParameterNames) {
                weightValues[p] = reader (p);
            }
        }

        public void WriteBuffers (WriteBuffer writer)
        {
            var vs = weightVectors.ToArray ();
            if (vs.Length == 0)
                return;
            var dev = vs[0].Value.Device;
            TryGetDataSource (dev)?.DownloadWeightsFromGpu ();
            foreach (var v in vs) {
                writer (v.Key, v.Value.ToSpan ());
            }
        }

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
                queue.Label = "WeightsLayer";
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
