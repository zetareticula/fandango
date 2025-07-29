use yew::prelude::*;
use yew_feather::{Grid, Play, Pause, StopCircle, Save, Folder, Plus, Settings};
use stylist::{style, yew::styled_component};
use web_sys::HtmlElement;

use crate::components::{
    block_palette::BlockPalette,
    workspace::Workspace,
    properties_panel::PropertiesPanel,
    status_bar::StatusBar,
};

/// Home page component
#[styled_component(HomePage)]
pub fn home_page() -> Html {
    // State for the workspace
    let workspace_ref = use_node_ref();
    
    // State for selected block
    let selected_block = use_state(|| None);
    
    // State for execution status
    let is_executing = use_state(|| false);
    let is_paused = use_state(|| false);
    
    // Handle block selection
    let on_block_select = {
        let selected_block = selected_block.clone();
        Callback::from(move |block_id| {
            selected_block.set(Some(block_id));
        })
    };
    
    // Handle execution control
    let on_execute = {
        let is_executing = is_executing.clone();
        Callback::from(move |_| {
            is_executing.set(true);
            // TODO: Start execution
        })
    };
    
    let on_pause = {
        let is_paused = is_paused.clone();
        Callback::from(move |_| {
            is_paused.set(!*is_paused);
            // TODO: Pause/resume execution
        })
    };
    
    let on_stop = {
        let is_executing = is_executing.clone();
        Callback::from(move |_| {
            is_executing.set(false);
            // TODO: Stop execution
        })
    };
    
    // Styles
    let container_style = style!(
        r#"
        display: flex;
        height: 100vh;
        width: 100vw;
        overflow: hidden;
        "#
    ).unwrap();
    
    let sidebar_style = style!(
        r#"
        width: 250px;
        background-color: #2d3748;
        color: white;
        padding: 1rem;
        display: flex;
        flex-direction: column;
        "#
    ).unwrap();
    
    let main_content_style = style!(
        r#"
        flex: 1;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        "#
    ).unwrap();
    
    let toolbar_style = style!(
        r#"
        background-color: #1a202c;
        color: white;
        padding: 0.5rem 1rem;
        display: flex;
        gap: 0.5rem;
        "#
    ).unwrap();
    
    let workspace_container_style = style!(
        r#"
        flex: 1;
        background-color: #1a1a1a;
        position: relative;
        overflow: auto;
        "#
    ).unwrap();
    
    html! {
        <div class={container_style}>
            {/* Left sidebar - Block palette */}
            <div class={sidebar_style}>
                <h2>{"Fandango"}</h2>
                <BlockPalette />
            </div>
            
            {/* Main content area */}
            <div class={main_content_style}>
                {/* Top toolbar */}
                <div class={toolbar_style}>
                    <button class="btn btn-sm btn-primary mr-2" onclick={on_execute} disabled={*is_executing}>
                        <Play size={16} class="mr-1" />
                        {"Run"}
                    </button>
                    <button class="btn btn-sm btn-warning mr-2" onclick={on_pause} disabled={!*is_executing}>
                        <Pause size={16} class="mr-1" />
                        {if *is_paused { "Resume" } else { "Pause" }}
                    </button>
                    <button class="btn btn-sm btn-danger mr-2" onclick={on_stop} disabled={!*is_executing}>
                        <StopCircle size={16} class="mr-1" />
                        {"Stop"}
                    </button>
                    <div class="flex-grow"></div>
                    <button class="btn btn-sm btn-secondary mr-2">
                        <Save size={16} class="mr-1" />
                        {"Save"}
                    </button>
                    <button class="btn btn-sm btn-secondary">
                        <Folder size={16} class="mr-1" />
                        {"Open"}
                    </button>
                </div>
                
                {/* Workspace area */}
                <div class={workspace_container_style} ref={workspace_ref}>
                    <Workspace on_block_select={on_block_select} />
                </div>
                
                {/* Status bar */}
                <StatusBar status={"Ready"} progress={0.0} />
            </div>
            
            {/* Right sidebar - Properties panel */}
            <PropertiesPanel selected_block={(*selected_block).clone()} />
        </div>
    }
}
