# Pipelines

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: yolo-pipelines
  namespace: openshift-gitops
spec:
  project: default
  destination:
    server: 'https://kubernetes.default.svc'
    namespace: yolo-uno-cards
  source:
    path: gitops/pipelines
    repoURL: https://github.com/alpha-hack-program/yolo-uno-cards.git
    targetRevision: main
    helm:
      parameters:
        - name: instanceName
          value: "yolo-uno-cards"
        - name: dataScienceProjectNamespace
          value: "yolo-uno-cards"
        - name: dataScienceProjectDisplayName
          value: "yolo-uno-cards"
  syncPolicy:
    automated:
      # prune: true
      selfHeal: true
```
