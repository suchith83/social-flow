export const chaosOptions = {
  duration: process.env.CHAOS_DURATION || 60,
  namespace: process.env.CHAOS_TARGET_NAMESPACE || "default",
  kubeconfig: process.env.KUBECONFIG || "~/.kube/config"
};
