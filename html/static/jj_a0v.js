function myonclick(i) {
 myrange = wvl_ranges[i-order_minmax[0]];
 g.updateOptions({dateWindow: myrange});
}

function myshowall() {
 myrange1 = wvl_ranges[0];
 myrange2 = wvl_ranges[order_minmax[1]-order_minmax[0]];
 myrange = [myrange2[0], myrange1[1]];
 g.updateOptions({dateWindow: myrange});
}

function dblClickV4(event, g, context) {
  restorePositioning(g);
}

function restorePositioning(g) {
  g.updateOptions({dateWindow: myrange});
}

var blockRedraw = false;

function mydraw(me, initial) {
                if (blockRedraw || initial) return;
                blockRedraw = true;
                var range = me.xAxisRange();
                for (var j = 0; j < 2; j++) {
                  if (gs[j] == me) continue;
                  gs[j].updateOptions( {
                    dateWindow: range
                  } );
                }
                blockRedraw = false;
              }

  gs = []

  g = new Dygraph(

    // containing div
    document.getElementById("graphdiv"),

    // CSV or path to a CSV file.
    first_filename,
      {
          connectSeparatedPoints: true,
	  // valueRange: [-value_max1/20.,value_max1],
	  dateWindow: wvl_ranges[0],
	  interactionModel : {
              'mousedown' : downV3,
              'mousemove' : moveV3,
              'mouseup' : upV3,
              'dblclick' : dblClickV4,
           },
	  drawGrid: false,
	  drawCallback: mydraw,
	  ylabel: 'Flattened A0V'
      }
  );
  gs.push(g)

  g2 = new Dygraph(

    // containing div
    document.getElementById("graphdiv2"),

    // CSV or path to a CSV file.
    second_filename,
      {
          connectSeparatedPoints: true,
	  // valueRange: [-value_max2,value_max2/20.],
	  dateWindow: wvl_ranges[0],
	  interactionModel : {
              'mousedown' : downV3,
              'mousemove' : moveV3,
              'mouseup' : upV3,
              'dblclick' : dblClickV4,
           },
	  drawGrid: false,
	  drawCallback: mydraw,
	  ylabel: 'Target/A0V'
      }
  );
  gs.push(g2)

  myonclick(order_minmax[0])
  // myshowall()
