#!/bin/sh

NAMESPACE=pipelines
SERVICE_ACCOUNT=pipeline

# Get RoleBindings
ROLE_BINDINGS=$(oc get rolebindings -n $NAMESPACE -o json | jq -r \
'.items[] | select(.subjects[]?.kind=="ServiceAccount" and .subjects[]?.name=="'$SERVICE_ACCOUNT'") | .roleRef.name')

# Get ClusterRoleBindings
CLUSTER_ROLE_BINDINGS=$(kubectl get clusterrolebindings -o json | jq -r \
'.items[] | select(.subjects[]?.kind=="ServiceAccount" and .subjects[]?.name=="'$SERVICE_ACCOUNT'" and .subjects[]?.namespace=="'$NAMESPACE'") | .roleRef.name')

echo "Roles bound to ServiceAccount $SERVICE_ACCOUNT in namespace $NAMESPACE:"
echo "$ROLE_BINDINGS"

echo "ClusterRoles bound to ServiceAccount $SERVICE_ACCOUNT in namespace $NAMESPACE:"
echo "$CLUSTER_ROLE_BINDINGS"
