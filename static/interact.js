function downV3(event, g, context) {
    context.initializeMouseDown(event, g, context);
    if (event.altKey || event.shiftKey) {
	Dygraph.startPan(event, g, context);
    } else {
	Dygraph.startZoom(event, g, context);
    }

}

function moveV3(event, g, context) {
  if (context.isPanning) {
    Dygraph.movePan(event, g, context);
  } else if (context.isZooming) {
    Dygraph.moveZoom(event, g, context);
  }
}

function upV3(event, g, context) {
  if (context.isPanning) {
    Dygraph.endPan(event, g, context);
  } else if (context.isZooming) {
    Dygraph.endZoom(event, g, context);
  }
}


function dblClickV4(event, g, context) {
  restorePositioning(g);
}

