using System;
namespace MetalTensors.Applications
{
    public class GanApplication
    {
        public Model Generator { get; }
        public Model Discriminator { get; }
        public Model Gan { get; }

        public GanApplication (Model generator, Model discriminator)
        {
            Generator = generator;
            Discriminator = discriminator;
            Gan = discriminator.Apply (generator);
        }
    }
}
