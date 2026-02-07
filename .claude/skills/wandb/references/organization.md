# W&B Organization: Teams, Projects & Registry

Structure your ML workflow with proper organization, access controls, and model management.

## Hierarchy Overview

```
Organization (root scope)
    └── Teams (business units, departments)
        └── Projects (individual ML experiments)
            └── Runs (training runs)
                └── Artifacts (datasets, models)
```

## Organizations

The root container for all W&B resources.

### Deployment Types
| Type | Description |
|------|-------------|
| **Multi-tenant Cloud** | Shared infrastructure, multiple orgs possible |
| **Dedicated Cloud** | Single-tenant in W&B's cloud (AWS/GCP/Azure) |
| **Self-Managed** | On-premise deployment |

### Organization Settings
- User management (invite/remove members)
- Team creation and administration
- Billing and usage tracking
- SSO/SAML configuration (Enterprise)

## Teams

A scope within an organization for collaboration.

### Creating Teams
1. Navigate to organization settings
2. Click "Create team"
3. Configure team name and settings

### Team Roles
| Role | Permissions |
|------|-------------|
| **Admin** | Full access, manage members, settings |
| **Member** | Create/edit runs, reports, artifacts |
| **View-Only** | View assets, comment on reports (Enterprise) |

### Privacy Settings
- **Hide from non-members**: Team invisible to outsiders
- **Private projects by default**: New projects are private
- **Restrict invites**: Only admins can invite members
- **Disable public sharing**: Block report sharing outside team

### Specifying Team in Code
```python
wandb.init(
    entity="my-team",     # Team name
    project="my-project",
)
```

## Projects

Container for related experiments and runs.

### Project Structure
```
Project
    ├── Runs (training runs)
    ├── Sweeps (hyperparameter searches)
    ├── Reports (documentation)
    ├── Artifacts (datasets, models)
    └── Automations (triggers, alerts)
```

### Visibility Modes
| Mode | Access |
|------|--------|
| **Private** | Team members only |
| **Public** | Anyone can view |
| **Open** | Anyone can contribute |

### Organizing Projects

**By Experiment Type**:
```
image-classification/
text-generation/
recommendation-system/
```

**By Model Architecture**:
```
resnet-experiments/
transformer-experiments/
gnn-experiments/
```

**By Dataset**:
```
imagenet-training/
coco-detection/
custom-data-v2/
```

### Groups Within Projects
```python
# Group related runs
wandb.init(
    project="hyperparameter-search",
    group="resnet50-learning-rates",
    job_type="train",
)
```

## Model Registry

Central repository for production-ready models.

### Registry Structure
```
Registry (organization-level)
    └── Collections (task-specific)
        └── Registered Models
            └── Versions (v0, v1, v2...)
                └── Aliases (latest, production, staging)
```

### Default Registries
- **Models**: Trained model artifacts
- **Datasets**: Dataset artifacts

### Linking Models to Registry
```python
with wandb.init() as run:
    # Create model artifact
    model = wandb.Artifact("my-model", type="model")
    model.add_file("model.pt")
    run.log_artifact(model)

    # Link to registry collection
    run.link_artifact(
        model,
        target_path="wandb-registry-model/image-classifier"
    )
```

### Using CLI
```bash
# Link existing artifact to registry
wandb artifact link \
    my-team/my-project/model:v3 \
    wandb-registry-model/image-classifier
```

### Managing Versions
```python
api = wandb.Api()

# List versions
collection = api.artifact_collection(
    "wandb-registry-model/image-classifier"
)
for version in collection.versions():
    print(f"{version.name}: {version.aliases}")

# Add alias
artifact = api.artifact("wandb-registry-model/image-classifier:v5")
artifact.aliases.append("production")
artifact.save()
```

## Access Control

### Project-Level Permissions
```
Private Project:
    - Team admins: Full access
    - Team members: Based on role
    - Others: No access

Public Project:
    - Everyone: View access
    - Team: Edit access
```

### Artifact-Level Permissions
Artifacts inherit project permissions. For sensitive artifacts:
1. Create separate private project
2. Use artifact references across projects

### Enterprise Features
- **Custom Roles**: Fine-grained permission composition
- **SSO/SAML**: Centralized authentication
- **Audit Logs**: Track all user actions
- **Data Retention Policies**: Automatic cleanup

## Best Practices

### Naming Conventions

**Teams**:
```
research-team
ml-platform
data-science
```

**Projects**:
```
<task>-<model>-<version>
image-classification-resnet-v2
nlp-bert-finetuning
```

**Artifacts**:
```
<type>-<name>
dataset-imagenet-train
model-resnet50-final
```

### Academic/Research Organization
```
Organization: university-ml-lab
    └── Team: paper-2024-attention
        ├── Project: ablation-studies
        ├── Project: main-experiments
        └── Project: baseline-comparisons
```

### Production ML Pipeline
```
Organization: company-name
    └── Team: ml-platform
        ├── Project: training-runs
        ├── Project: evaluation-runs
        └── Registry:
            ├── Collection: prod-models
            └── Collection: staging-models
```

### Governance Workflow
```
1. Train models → my-team/training-project
2. Evaluate → my-team/eval-project
3. Promote to registry → wandb-registry-model/collection
4. Tag for staging → :staging alias
5. Tag for production → :production alias
```

## Automations

### Run Completion Triggers
```python
# In W&B UI: Automations → Create Automation
# Trigger: Run completes with val_accuracy > 0.95
# Action: Link to registry collection
```

### Slack Notifications
```python
# Alert on metric threshold
wandb.alert(
    title="High Accuracy Achieved",
    text=f"val_accuracy: {accuracy:.2%}",
)
```

### Registry Webhooks
Configure in UI:
1. Navigate to Registry collection
2. Click "Connect Slack" or configure webhook
3. Receive notifications on new versions

## API Access

### Public API
```python
import wandb

api = wandb.Api()

# List all projects in team
projects = api.projects(entity="my-team")

# List all runs in project
runs = api.runs("my-team/my-project")

# Filter runs
runs = api.runs(
    "my-team/my-project",
    filters={"config.lr": 0.001}
)

# Access run data
for run in runs:
    print(run.name, run.summary["accuracy"])
```

### Service Accounts (Enterprise)
For CI/CD pipelines:
```bash
export WANDB_API_KEY=<service-account-key>
wandb login --relogin
```

## Common Anti-Patterns

| DON'T | DO |
|-------|-----|
| One project for everything | Separate projects by experiment type |
| Personal account for team work | Create team, set as entity |
| Hardcode entity in code | Use environment variables |
| Skip registry | Link production models |
| Share API keys | Use service accounts |
| Public projects with sensitive data | Use private projects |
