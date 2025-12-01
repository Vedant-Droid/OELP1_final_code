bl_info = {
    "name": "Cricket Trajectory AI",
    "author": "Akshat",
    "version": (1, 1),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Cricket AI",
    "description": "Generates cricket ball trajectories using external AI model",
    "category": "Animation",
}

import bpy # type: ignore
import subprocess
import json
import os

class CricketAIProperties(bpy.types.PropertyGroup):
    python_path: bpy.props.StringProperty(
        name="Python Path",
        description="Path to python executable in tf_venv",
        default="/home/akshat/code/OELP/Final_code_base/tf_venv/bin/python", # Update this!
        subtype='FILE_PATH'
    ) # type: ignore
    
    project_path: bpy.props.StringProperty(
        name="Project Path",
        description="Folder containing generate_shot.py",
        default="/home/akshat/code/OELP/Final_code_base/", # Update this!
        subtype='DIR_PATH'
    ) # type: ignore
    
    v_cat: bpy.props.EnumProperty(
        name="Velocity",
        items=[
            ('Low', "Low (< 22 m/s)", ""),
            ('Medium', "Medium (22-26 m/s)", ""),
            ('High', "High (> 26 m/s)", ""),
        ],
        default='Medium'
    ) # type: ignore
    
    w_mag_cat: bpy.props.EnumProperty(
        name="Spin Amount",
        items=[
            ('Low', "Low", ""),
            ('Medium', "Medium", ""),
            ('High', "High", ""),
        ],
        default='Medium'
    ) # type: ignore
    
    w_angle_cat: bpy.props.EnumProperty(
        name="Spin Direction",
        items=[
            ('Negative_Spin', "Leg/In (Neg)", "Spin to Left"),
            ('Neutral_Spin', "Straight", "No side spin"),
            ('Positive_Spin', "Off/Out (Pos)", "Spin to Right"),
        ],
        default='Neutral_Spin'
    ) # type: ignore

class OBJECT_OT_GenerateTrajectory(bpy.types.Operator):
    """Generate Trajectory using AI"""
    bl_idname = "cricket.generate_trajectory"
    bl_label = "Generate Trajectory"
    
    def execute(self, context):
        props = context.scene.cricket_props
        ball = context.active_object
        target = bpy.data.objects.get("land_point")
        
        if not ball:
            self.report({'ERROR'}, "No active ball object selected")
            return {'CANCELLED'}
        
        if not target:
            self.report({'ERROR'}, "No object named 'land_point' found")
            return {'CANCELLED'}
            
        # 1. Prepare Paths
        script_path = os.path.join(props.project_path, "generate_shot.py")
        output_path = os.path.join(props.project_path, "trajectory.json")
        
        if not os.path.exists(script_path):
            self.report({'ERROR'}, f"Script not found at {script_path}")
            return {'CANCELLED'}

        # 2. Build Command (Theta Removed)
        cmd = [
            props.python_path,
            script_path,
            "--start", str(ball.location.x), str(ball.location.y), str(ball.location.z),
            "--target", str(target.location.x), str(target.location.y),
            "--v_cat", props.v_cat,
            "--w_mag_cat", props.w_mag_cat,
            "--w_angle_cat", props.w_angle_cat,
            "--output", output_path
        ]
        
        # 3. Execute Subprocess
        try:
            print("Running AI...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.report({'ERROR'}, f"AI Error: {result.stderr}")
                print(result.stderr)
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Execution Failed: {e}")
            return {'CANCELLED'}
            
        # 4. Read JSON and Animate
        if not os.path.exists(output_path):
            self.report({'ERROR'}, "No JSON output generated")
            return {'CANCELLED'}
            
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            self.report({'ERROR'}, "Failed to decode JSON output")
            return {'CANCELLED'}
            
        if "error" in data:
            self.report({'ERROR'}, data['error'])
            return {'CANCELLED'}
            
        self.animate_ball(ball, data['trajectory'])
        self.report({'INFO'}, "Trajectory Generated Successfully!")
        return {'FINISHED'}

    def animate_ball(self, obj, trajectory):
        # Clear existing animation
        obj.animation_data_clear()
        obj.animation_data_create()
        action = bpy.data.actions.new(name="Trajectory")
        obj.animation_data.action = action
        
        # FPS assumption (Physics sim ran at 60sps, Blender default is 24fps)
        # We map physics samples directly to frames. 
        # For real-time speed, ensure Blender Render Frame Rate is set to 60fps.
        
        for point in trajectory:
            frame = int(point['sample']) # Assuming sample 0, 1, 2...
            pos = point['position']
            
            # Update location
            obj.location = (pos[0], pos[1], pos[2])
            obj.keyframe_insert(data_path="location", frame=frame)
            
        # Set End Frame
        bpy.context.scene.frame_end = int(trajectory[-1]['sample'])

class PT_CricketPanel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Cricket AI"
    bl_idname = "PT_CricketPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Cricket AI"

    def draw(self, context):
        layout = self.layout
        props = context.scene.cricket_props

        layout.label(text="Environment Config")
        layout.prop(props, "python_path")
        layout.prop(props, "project_path")
        
        layout.separator()
        layout.label(text="Shot Parameters")
        
        row = layout.row()
        row.prop(props, "v_cat")
        
        row = layout.row()
        row.prop(props, "w_mag_cat")
        
        row = layout.row()
        row.prop(props, "w_angle_cat")
        
        # Removed Theta Cat UI
        
        layout.separator()
        layout.operator("cricket.generate_trajectory", icon='PHYSICS')

def register():
    bpy.utils.register_class(CricketAIProperties)
    bpy.utils.register_class(OBJECT_OT_GenerateTrajectory)
    bpy.utils.register_class(PT_CricketPanel)
    bpy.types.Scene.cricket_props = bpy.props.PointerProperty(type=CricketAIProperties)

def unregister():
    bpy.utils.unregister_class(PT_CricketPanel)
    bpy.utils.unregister_class(OBJECT_OT_GenerateTrajectory)
    bpy.utils.unregister_class(CricketAIProperties)
    del bpy.types.Scene.cricket_props

if __name__ == "__main__":
    register()