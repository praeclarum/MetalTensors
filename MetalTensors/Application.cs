using System;

namespace MetalTensors
{
    public abstract class Application
    {
        public virtual int Run (string[] args)
        {
            var r = ParseArgs (args);
            if (r != 0)
                return r;
            try {
                //Train ();
                return 0;
            }
            catch (Exception ex) {
                LogError (ex);
                return 2;
            }
        }

        public static int Run (Type appType, string[] args)
        {
            var app = (Application)Activator.CreateInstance (appType);
            return app.Run (args);
        }

        protected virtual int ParseArgs (string[] args)
        {
            try {
                return 0;
            }
            catch (Exception ex) {
                LogError (ex);
                return 1;
            }
        }

        protected virtual void LogError (Exception ex)
        {
            Console.WriteLine (ex);
        }
    }
}
