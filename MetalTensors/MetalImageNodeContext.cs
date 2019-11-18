using System;
using Metal;

namespace MetalTensors
{
    public class MetalImageNodeContext
    {
        public string CacheKey { get; }
        public string Label { get; }
        public bool IsTraining { get; }
        public IMTLDevice Device { get; }

        public MetalImageNodeContext(string label, bool isTraining, IMTLDevice device)
        {
            Label = label;
            IsTraining = isTraining;
            Device = device;
            CacheKey = $"{Label} + {device.Name}:{device.Handle} + {Guid.NewGuid ()}";
        }
    }
}
