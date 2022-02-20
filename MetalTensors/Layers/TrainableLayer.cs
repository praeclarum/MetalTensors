using System;

namespace MetalTensors.Layers
{
    public abstract class TrainableLayer : Layer
    {
        bool isTrainable = true;

        public override bool IsTrainable {
            get => isTrainable;
            set => isTrainable = value;
        }

        protected TrainableLayer (string? name = null, bool isTrainable = true)
            : base (name)
        {
            this.isTrainable = isTrainable;
        }

        public override Config Config => base.Config.Update (new Config {
            { "isTrainable", IsTrainable },
        });
    }
}
