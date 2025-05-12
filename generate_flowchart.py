import graphviz

# Create a directed graph
graph = graphviz.Digraph(
    name='Object Detection Model Building Workflow',
    comment='Comprehensive workflow for object detection model building',
    format='png',
    engine='dot'
)

# Set graph attributes
graph.attr(
    rankdir='TB',  # Top to Bottom layout
    size='11,16',  # Size in inches
    dpi='300',     # Resolution
    fontname='Arial',
    fontsize='16',
    bgcolor='white',
    nodesep='0.5',
    ranksep='0.75',
)

# Define node and edge attributes for different sections
graph.attr('node', shape='box', style='filled', fontname='Arial', fontsize='12', margin='0.2')
graph.attr('edge', fontname='Arial', fontsize='10', arrowsize='0.8')

# Define colors for different sections
project_overview_color = '#A3D9FF'  # Light Blue
implementation_color = '#FFD9A3'    # Light Orange
deliverables_color = '#A3FFA3'      # Light Green
ai_integration_color = '#FFA3D9'    # Light Pink
decision_color = '#D9D9D9'          # Light Gray

# Create clusters (subgraphs) for each main section
with graph.subgraph(name='cluster_project_overview') as c:
    c.attr(label='Project Overview', style='filled', color=project_overview_color, fontsize='14')
    
    # Initial setup
    c.node('setup', 'Initial Setup &\nPrerequisites', shape='box', color='#6495ED')
    
    # CNN Backbone Selection (Decision Point)
    c.node('backbone_decision', 'CNN Backbone\nSelection', shape='diamond', color=decision_color)
    c.node('resnet', 'ResNet (50, 101)', color=project_overview_color)
    c.node('vgg16', 'VGG16', color=project_overview_color)
    c.node('mobilenet', 'MobileNet', color=project_overview_color)
    c.node('efficientnet', 'EfficientNet', color=project_overview_color)
    c.node('densenet', 'DenseNet', color=project_overview_color)
    
    # Detection Approach (Decision Point)
    c.node('detection_approach_decision', 'Detection Approach\nSelection', shape='diamond', color=decision_color)
    c.node('ssd', 'Single Shot Detector\n(SSD)', color=project_overview_color)
    c.node('fpn', 'Feature Pyramid\nNetwork (FPN)', color=project_overview_color)
    c.node('rpn', 'Region Proposal\nNetwork (RPN)', color=project_overview_color)
    c.node('yolo', 'YOLO-style\nDetection Head', color=project_overview_color)
    
    # Dataset Selection (Decision Point)
    c.node('dataset_decision', 'Dataset\nSelection', shape='diamond', color=decision_color)
    c.node('coco', 'COCO', color=project_overview_color)
    c.node('pascal_voc', 'Pascal VOC', color=project_overview_color)
    c.node('open_images', 'Open Images', color=project_overview_color)
    c.node('kitti', 'KITTI', color=project_overview_color)
    c.node('custom', 'Custom Datasets', color=project_overview_color)

with graph.subgraph(name='cluster_implementation') as c:
    c.attr(label='Implementation Flow', style='filled', color=implementation_color, fontsize='14')
    
    c.node('backbone_loading', 'Backbone Loading\n& Modification', color='#FF9E5E')
    c.node('detection_head', 'Detection Head\nImplementation', color='#FF9E5E')
    c.node('training_pipeline', 'Training Pipeline\nSetup', color='#FF9E5E')
    c.node('model_eval', 'Model Evaluation\nProcess', color='#FF9E5E')

with graph.subgraph(name='cluster_deliverables') as c:
    c.attr(label='Deliverables Tracking', style='filled', color=deliverables_color, fontsize='14')
    
    c.node('source_code', 'Source Code\nGeneration', color='#5ECC5E')
    c.node('model_weights', 'Model Weights\nManagement', color='#5ECC5E')
    c.node('eval_metrics', 'Evaluation Metrics\nCalculation', color='#5ECC5E')
    c.node('demo_prep', 'Demo\nPreparation', color='#5ECC5E')
    c.node('report_writing', 'Experience Report\nWriting', color='#5ECC5E')

with graph.subgraph(name='cluster_ai_integration') as c:
    c.attr(label='AI Integration Points', style='filled', color=ai_integration_color, fontsize='14')
    
    c.node('code_gen_assistance', 'Code Generation\nAssistance', color='#E64980')
    c.node('debugging_support', 'Debugging\nSupport', color='#E64980')
    c.node('documentation_help', 'Documentation\nHelp', color='#E64980')

# Add connections between nodes
# Project Overview connections
graph.edge('setup', 'backbone_decision')

# Backbone decision connections
graph.edge('backbone_decision', 'resnet', label='Option 1')
graph.edge('backbone_decision', 'vgg16', label='Option 2')
graph.edge('backbone_decision', 'mobilenet', label='Option 3')
graph.edge('backbone_decision', 'efficientnet', label='Option 4')
graph.edge('backbone_decision', 'densenet', label='Option 5')

# Multiple options converge to next step
graph.edge('resnet', 'detection_approach_decision')
graph.edge('vgg16', 'detection_approach_decision')
graph.edge('mobilenet', 'detection_approach_decision')
graph.edge('efficientnet', 'detection_approach_decision')
graph.edge('densenet', 'detection_approach_decision')

# Detection approach connections
graph.edge('detection_approach_decision', 'ssd', label='Option 1')
graph.edge('detection_approach_decision', 'fpn', label='Option 2')
graph.edge('detection_approach_decision', 'rpn', label='Option 3')
graph.edge('detection_approach_decision', 'yolo', label='Option 4')

# Multiple options converge to next step
graph.edge('ssd', 'dataset_decision')
graph.edge('fpn', 'dataset_decision')
graph.edge('rpn', 'dataset_decision')
graph.edge('yolo', 'dataset_decision')

# Dataset decision connections
graph.edge('dataset_decision', 'coco', label='Option 1')
graph.edge('dataset_decision', 'pascal_voc', label='Option 2')
graph.edge('dataset_decision', 'open_images', label='Option 3')
graph.edge('dataset_decision', 'kitti', label='Option 4')
graph.edge('dataset_decision', 'custom', label='Option 5')

# Connect to Implementation Flow
graph.edge('coco', 'backbone_loading')
graph.edge('pascal_voc', 'backbone_loading')
graph.edge('open_images', 'backbone_loading')
graph.edge('kitti', 'backbone_loading')
graph.edge('custom', 'backbone_loading')

# Implementation Flow connections
graph.edge('backbone_loading', 'detection_head')
graph.edge('detection_head', 'training_pipeline')
graph.edge('training_pipeline', 'model_eval')
# Feedback loop from evaluation back to training
graph.edge('model_eval', 'training_pipeline', label='Refinement\nLoop', style='dashed')

# Deliverables connections
graph.edge('backbone_loading', 'source_code')
graph.edge('detection_head', 'source_code')
graph.edge('training_pipeline', 'model_weights')
graph.edge('model_eval', 'eval_metrics')
graph.edge('eval_metrics', 'demo_prep')
graph.edge('model_weights', 'demo_prep')
graph.edge('demo_prep', 'report_writing')

# AI Integration connections
graph.edge('code_gen_assistance', 'source_code', style='dashed', color='#E64980')
graph.edge('debugging_support', 'training_pipeline', style='dashed', color='#E64980')
graph.edge('debugging_support', 'model_eval', style='dashed', color='#E64980')
graph.edge('documentation_help', 'report_writing', style='dashed', color='#E64980')

# Save the graph
output_path = 'D:\\Object Detection Model Building\\project_workflow'
graph.render(output_path, view=False, cleanup=True)

print(f"Flow chart created successfully at: {output_path}.png")

