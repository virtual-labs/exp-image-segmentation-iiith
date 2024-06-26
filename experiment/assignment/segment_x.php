<?php
session_start();

if($_SESSION["piping"]!=99) {
	if(isset($_GET["source"]) && isset($_GET["id"])) {
		$_SESSION["currentS"]=$_GET["source"]; 
		$_SESSION["id"]=$_GET["id"]; 
	} else {
		$_SESSION["id"]=time();
		$_SESSION["state"]=0;
	}
}

if(isset($_GET["assess"])) {
	$_SESSION["currentS"]=1;
	$_SESSION["id"]=$_GET["assess"];
	$_SESSION["state"]=1;
	}

?> 


<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">

<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<script class='gtm'>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src='https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);})(window,document,'script','dataLayer','GTM-W59SWTR');</script>


<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1" />
<!-- This file was originally generated at http://psd2cssonline.com on September 16, 2010, 10:04 am -->
<!-- psd2css Online version 1.85 -->

<title>Virtual Lab in Image Processing</title>


<!-- Some META tags to help with Search Engine Optimization.  Please 
note however that META tags are NOT a magic bullet to get your web page
to the top of search engine rankings.  They are just part of that effort.  You 
can get more information by googling SEO or visiting the psd2css Online forums. -->
<meta name="description" content="Put the description of this page here" />
<meta name="keywords" content="Put keywords for this page here separated by commas" />
<meta name="generator" content="psd2css Online - Dynamic Web Pages from your Photoshop Design in seconds" />


<!-- The CSS Reset from Eric Meyers -->
<!-- <link rel="stylesheet" type="text/css" href="cssreset.css" media="screen" /> -->


<!-- The Primary External CSS style sheet. -->
<link rel="stylesheet" type="text/css" href="css/psd2css.css" media="screen" />


<!-- We use the jquery javascript library for DOM manipulation and
some javascript tricks.  We serve the script from Google because it's
faster than most ISPs.  Get more information and documentation
at http://jquery.com <script type="text/javascript" src="http://ajax.googleapis.com/ajax/libs/jquery/1.4/jquery.min.js"> -->
<script type="text/javascript" src="js/jquery-1.4.2.min.js"> </script> 


<!-- All the javascript generated for your design is in this file -->
<script type="text/javascript" src="js/psd2css.js"></script>

<!-- For Jquery UI-->
<script type="text/javascript" src="js/jquery-ui-1.8.4.custom.min.js"></script>
<link rel="stylesheet" type="text/css" href="css/dark-hive/jquery-ui-1.8.4.custom.css">


<!--http://www.cssmenumaker.com/builder/menu_info.php?menu=057-->
<link type="text/css" rel="StyleSheet" href="menu/menu_style.css" />


<!--for ImgSelect - http://odyniec.net/projects/imgareaselect/-->
<link rel="stylesheet" type="text/css" href="css/imgareaselect-default.css" />  
<script type="text/javascript" src="js/jquery.imgareaselect.js"></script> 


<!--for Flot -->
<script type="text/javascript" src="js/jquery.flot.js"></script> 

<!--for COntent Slider - Jquery Slider/-->
 <link rel="stylesheet" type="text/css" href="css/jquery-slider.css">


<script type="text/javascript">

function blinker(i){
if(buttId!="cancel" && i>0) {
$($(buttId),"#Layer-2").toggleClass("ui-state-hover");
setTimeout("blinker("+(i-1)+")",500);


}
}

function set_state(state) {
switch (state) {
case 0: $("#nextBox").html("Start the Experiment by clicking on 'Select Image' and selecting an Input Image from the Mosaic");
buttId="button1"; 
blinker(6);
break;
case 1: $("#nextBox").html("Select appropriate parameters and click on 'Run'.");
buttId="button2";
blinker(6);
break;
//case 2: $("#nextBox").html("Observe the hostogram, try making changes to the bin size and observe the trends.");
case 2: $("#nextBox").html("Note down the results");
break;
case 3: $("#nextBox").html("Click on each of the four subgraphs to view the full size.");

}
jstate=state;

}


  function toggleMosaic()    {
  var ias = $('#Mosaic').imgAreaSelect({ instance: true });
      if($("#Mosaic").is(":visible")) {
        ias.setOptions({ hide: true });
        $("#Mosaic").hide("slow");
      } else {
        $("#Mosaic").show("slow", function() {
          ias.setOptions({ show: true });
          ias.update();
		$(".reGrow").hide();  
          
        });
      }

  }
  
  
function selectP (event) {
pos_x = parseInt((event.pageX-document.getElementById("Layer-1").offsetLeft-document.getElementById("threSelect").offsetLeft)*64/300);
if(pos_x>63) { pos_x=63;}

if(turn=="start" && sing!="yes") {
endx=pos_x;
turn="end";
} else {
startx=pos_x;
turn="start"; }

$("#"+turn).css("left",pos_x*300/64 +document.getElementById("Layer-4").offsetLeft-15) ;
}

function pointer (event) {
var num_blocks=150;
pos_x = parseInt((event.pageX-document.getElementById("Layer-1").offsetLeft-document.getElementById("Layer-4").offsetLeft)*num_blocks/281);
pos_y = parseInt((event.pageY-document.getElementById("Layer-1").offsetTop-document.getElementById("Layer-4").offsetTop)*num_blocks/281);
if(pos_x>num_blocks-1) { pos_x=num_blocks-1;}
if(pos_y>num_blocks-1) { pos_y=num_blocks-1;}

startx2=pos_x;
starty2=pos_y;
$("#st").html("x = " + (pos_x*2) + "  y = " + (pos_y*2));
$("#start2").css("left",pos_x*281/num_blocks +document.getElementById("Layer-4").offsetLeft-2) ;
$("#start2").css("top",pos_y*281/num_blocks +document.getElementById("Layer-4").offsetTop-2) ;

}

</script>






<script tyre="text/javascript">


function set_crop() {
    set_state(1);
		$(".reGrow").show();


    var ias = $('#Mosaic').imgAreaSelect({ instance: true });
	var sel=ias.getSelection();

    //alert('sdfsd');

    var aurl='x=' + sel.x1 + '&y=' + sel.y1 + '&w=' + sel.width + '&h=' + sel.height + '&exp=segment';


    $.ajax({url: "crop.php" , type: "POST", data: aurl, async: false, dataType: "html", success: function(data) {
		$("#Layer-4").html(data);		


	       
	     }});

}


</script>




<script type="text/javascript">
function doTransform(){

    	if(jstate == 0) {
	    alert("First click on Mosaic and select the Input Image.");
            return;

    	}
		var args;
		var checkBoxes = $("input[name=choice]");

		$.each(checkBoxes, function() {
					if ($(this).attr('checked')){
				args = $(this).attr('value');

				}	
			});

		if(args == 1) {
		
		checkBoxes = $("input[name=option1]");

		$.each(checkBoxes, function() {
					if ($(this).attr('checked')){
				args = $(this).attr('value');

				}	
			});

		if(jstate!=2&&args!="Auto") {

			var aurl="trns.php?opn=histo&mode=1&bins=64";
			$.ajax({ url: aurl, type: "GET", async: false, dataType: "html", success: function(data){
				$(".content-conveyor", $("#sliderContent")).html(data);	
			}});
			
			$("#sliderContent").append('<div id="slider"></div>');
			
			var conveyor2 = $(".content-conveyor", $("#sliderContent")),
			item2 = $(".item", $("#sliderContent"));
			conveyor2.css("left","0px");
			
			//set length of conveyor
			conveyor2.css("width", item2.length * parseInt(item2.css("width")));
			conveyor2.css("left","-" + ((item2.length * parseInt(item2.css("width"))) - parseInt($(".viewer", $("#sliderContent")).css("width"))) + "px");
			$("img",$("#threSelect")).attr("src",$("img",$(".item:last")).attr("src")); 
			$("#threSelect").show();
			$("#navig").hide();
			set_state(2);
			$("#start").show();
			if(sing=="no") {
				$("#end").show();
			}
			$(".notMan").attr("disabled","true");
		return;
		} else {
					
			if(jstate==2) {
					
					$(".notMan").removeAttr("disabled");
					$("#threSelect").hide();
					$("#navig").show();
					set_state(1);
					$("#start").hide();
					$("#end").hide();
							
					args += "&endx=" + endx;
					args += "&startx=" + startx;

					if(!($('#Mosaic').is(":visible"))) {
					var aurl = "trns.php?opn=segment&args=" + args;
					   $("#slider").remove();	
					   $.ajax({ url: aurl, type: "GET", async: false, dataType: "html", success: function(data){
					$(".content-conveyor", $("#sliderContent")).html(data);	

					}});
					$("#sliderContent").append('<div id="slider"></div>');

					
					var conveyor2 = $(".content-conveyor", $("#sliderContent")),
					item2 = $(".item", $("#sliderContent"));

					
					//set length of conveyor
					conveyor2.css("width", item2.length * parseInt(item2.css("width")));
						conveyor2.css("left","-" + ((item2.length * parseInt(item2.css("width"))) - parseInt($(".viewer", $("#sliderContent")).css("width"))) + "px");

							
					//config
					var sliderOpts = {
					  max: (item2.length * parseInt(item2.css("width"))) - parseInt($(".viewer", $("#sliderContent")).css("width")),
					  value:  (item2.length * parseInt(item2.css("width"))) - parseInt($(".viewer", $("#sliderContent")).css("width")),
					  slide: function(e, ui) { 
						conveyor2.css("left", "-" + ui.value + "px");
					  }
					};
					
					//create slider
					$("#slider").slider(sliderOpts);

					set_state(4);

					   

					$("img",$("#Layer-5")).attr("src",$("img",$(".item:last")).attr("src")); 

					   
					}
					return;
			}
		}
	}
	
	if(args!="Auto") {
		if($("#regionG").attr("checked")) {
			checkBoxes = $("input[name=option3]");

			$.each(checkBoxes, function() {
				if ($(this).attr('checked')){
					args += "&option2=1&option3=" + $(this).attr('value');
					
				}	
			});
		} else {
			checkBoxes = $("input[name=option4]");

			$.each(checkBoxes, function() {
				if ($(this).attr('checked')){
					args += "&option2=2&option3=" + $(this).attr('value');
					
				}	
			});
		}
		args+="&y="+ (startx2*2) + "&x=" + (starty2*2);
		
		if($("#regionG").attr("checked")){ 
		args+="&dev="+$( "#slider-Dev" ).slider( "option", "value" );
		} else {
		args+="&dev="+$( "#slider-Dev2" ).slider( "option", "value" );

		}
		
	}
	if(!($('#Mosaic').is(":visible"))) {
	var aurl = "trns.php?opn=segment&args=" + args;
	   $("#slider").remove();	
	   $.ajax({ url: aurl, type: "GET", async: false, dataType: "html", success: function(data){
	$(".content-conveyor", $("#sliderContent")).html(data);	

	}});
	$("#sliderContent").append('<div id="slider"></div>');

	
	var conveyor2 = $(".content-conveyor", $("#sliderContent")),
	item2 = $(".item", $("#sliderContent"));

	
	//set length of conveyor
	conveyor2.css("width", item2.length * parseInt(item2.css("width")));
		conveyor2.css("left","-" + ((item2.length * parseInt(item2.css("width"))) - parseInt($(".viewer", $("#sliderContent")).css("width"))) + "px");

			
	//config
	var sliderOpts = {
	  max: (item2.length * parseInt(item2.css("width"))) - parseInt($(".viewer", $("#sliderContent")).css("width")),
	  value:  (item2.length * parseInt(item2.css("width"))) - parseInt($(".viewer", $("#sliderContent")).css("width")),
	  slide: function(e, ui) { 
		conveyor2.css("left", "-" + ui.value + "px");



	  }
	};
	
	//create slider
	$("#slider").slider(sliderOpts);

	set_state(4);

	   

	$("img",$("#Layer-5")).attr("src",$("img",$(".item:last")).attr("src")); 

	   
	}
	
}

  $(document).ready (function() {
  startx=18;
  endx=33;
  turn="end";
  sing="yes";
  buttId="button1";
    <?php if(isset($_SESSION["currentS"]) || isset($_GET["assess"])) {
	echo "set_state(1);"; } else { echo "set_state(0);"; ;
	} 
	
	?>
  
  $(".imageFull").hide();
  $("#popUp").hide();
     $("button1","#Layer-2").button();
  $("button1","#Layer-2").click( function() {
  toggleMosaic();
  });

  $("button2","#Layer-2").button();
  $("button2","#Layer-2").click ( function () {
  doTransform();
  });
  
  $("button3").button();
  $("button3").click ( function () {
  doTransform();
  });

  $("button4","#Layer-2").button();
  $("button4","#Layer-2").click ( function () {
 <?php if(!isset($_SESSION["currentS"]) && !isset($_GET["assess"])) { echo ' set_state(0);'; } 
		else { echo 'set_state(1);'; }?>
  $("dims","#imgInfo").html("____ x ____ ");
  
  $("img",$("#Layer-5")).attr("src","images/Layer-5.jpg");  
<?php  if( $_SESSION["piping"]!=99 ) { ?> 
 $("img",$("#Layer-4")).attr("src","<?php if(isset($_GET["assess"])) { echo 'images/temp/'.$_SESSION["id"].'_'.$_SESSION["currentS"].'_result.png'; } 
		else { echo 'images/Layer-4.jpg'; } ?>"); <?php } ?>
$("#sliderContent").replaceWith('      <div id="sliderContent" class="ui-corner-all">		<div class="viewer ui-corner-all">	  <div class="content-conveyor ui-helper-clearfix">	    <div class="item">	      <h2><?php if(isset($_GET["assess"])) {	  echo 'Assessment';	  } else { echo 'Start'; } ?></h2>	      <img src="<?php if(isset($_GET["assess"])) {	  echo 'images/temp/'.$_SESSION["id"].'_'.$_SESSION["currentS"].'.png';	  } else { echo 'images/Mosaichisto.png'; } ?>" alt="picture" width="140px" height="140px"/>	      <dl class="details ui-helper-clearfix">		<dt><?php if(isset($_GET["assess"])) {	  echo 'This is the Input Image. Perform Operations to get the above result.';	  } else { echo 'Select a portion of the Mosaic on the Right and Transform... '; } ?></dt>	      </dl>	    </div>	  </div>	</div>	<div id="slider"></div>      </div>');
});
 
  
  $("#Mosaic").imgAreaSelect({ 
  handles: true, 
  movable: true,
  persistent: true,
  resizable: false,
  x1: 300, y1: 300, x2: 600, y2: 600,
  hide: false,
  imageHeight: 900,
  imageWidth: 900,
  });
  toggleMosaic();
  
  $('#placeholder').hide();
  
  $("button7","#Layer-2").button();
	$("button7","#Layer-2").click( function() {
		var answer = confirm("Retain Changes?")
		if (answer) {
			if(jstate>1) {
				var aurl="item=" + $(".item:last").html();

				$.ajax({url: "write.php" , type: "POST", data: aurl, async: false, dataType: "html", success: function(data) {
				window.location.href="piping.php";

				}});
			}
		}
		else {
			window.location.href="piping.php";
		}
  });
  
    	<?php 
			if(isset($_GET["args"])) {
			if($_GET["args"] != "regn") {
				echo '
				checkBoxes = $("input[name=option1]");

				$.each(checkBoxes, function() {
					if ($(this).attr("value")=="'.$_GET["args"].'") {
						$(this).click();
					}	
				}); ';
			
			
			} else {
				echo '

				$("#regionGrowing").click();

				';
			}
			}
	?>
  
});



</script>
</head>

<body>

  <!-- This is 'Backgound_bkgnd_center_jpg' -->
  <div id="Layer-1" class="Backgound_bkgnd_center_jpg"  >

    <div id="para_pop"> </div>

    <!-- This is 'Mosaic_jpg' -->
    <div id="Mosaic" class="Mosaic_jpg"  >
      <img src="images/Mosaicsegment.png" width="513" height="513" alt="Mosaic" /></div>

    <!-- This is 'Thumbnails_jpg' -->
    <div id="Layer-7" class="Thumbnails_jpg"  >
      <div id="sliderContent" class="ui-corner-all">	
	<div class="viewer ui-corner-all">
	  <div class="content-conveyor ui-helper-clearfix">
	    <div class="item">
	      <h2><?php if(isset($_GET["assess"])) {
	  echo 'Assessment';
	  } else { echo 'Start'; } ?></h2>
	      <img src="<?php if(isset($_GET["assess"])) {
	  echo 'images/temp/'.$_SESSION["id"].'_'.$_SESSION["currentS"].'.png';
	  } else { echo 'images/Mosaicsegment.png'; } ?>" alt="picture" width="140px" height="140px"/>
	      <dl class="details ui-helper-clearfix">
		<dt><?php if(isset($_GET["assess"])) {
	  echo 'This is the Input Image. Perform Operations to get the above result.';
	  } else { echo 'Select a portion of the Mosaic on the Right and Transform... '; } ?></dt>
	      </dl>
	    </div>
	  </div>
	</div>
	<div id="slider"></div>
      </div>
     </div>

   <!-- This Is 'outputLayer_jpg' -->
    <div id="Layer-5" class="outputLayer_jpg"  >
      <img src="images/Layer-5.jpg" alt="outputLayer" width="281px" height="281px"/></div>

    <!-- This is 'InputLayer_jpg' -->
    <div id="Layer-4" class="InputLayer_jpg"  onclick="pointer(event)">
      <img src="<?php if(isset($_GET["source"])  || $_SESSION["piping"]==99) {
	  echo 'images/temp/'.$_SESSION["id"].'_'.$_SESSION["currentS"].'.png';
	  } else { if(isset($_GET["assess"])) { echo 'images/temp/'.$_SESSION["id"].'_'.$_SESSION["currentS"].'_result.png'; } 
		else { echo 'images/Layer-4.jpg'; }} ?>" width="281" height="281" alt="InputLayer"/></div>

	<div id="threSelect" style="position: absolute; border: 1px white solid; left: 310px; width: 300px; top: 111px; padding: 5px; image-rendering: -moz-crisp-edges; z-index: 23; display:none; background-color: #444444;">   
	<img src="#" onclick="selectP(event)" /> 
	<br/><br/>
	<button3 style="margin-left:auto;">Proceed</button3>
	</div> 

	<div id="start" style="position: absolute; left: 393px; top: 117px; z-index: 25; display:none; ">
	<img src="segment/column.png"/>
	</div>

	<div id="end" style="position: absolute; left: 463px; top: 117px; z-index: 26; display:none;">
	<img src="segment/column.png"/>
	</div>
	
	<div id="seedCont" style="display:none;">
	<div id="start2" class="reGrow" style="position: absolute; left: 393px; top: 192px; z-index: 30; display:none;">
	<img src="diff/end2.png"/>
	</div>
	</div>
		
    <!-- This is 'TopBar_jpg' -->
    <div id="Layer-3" class="TopBar_jpg"  >
      <img src="images/Layer-3.jpg" width="894" height="96" alt="TopBar" />
      <!-- This is 'IIIT' -->
      <div id="Layer-6" class="IIIT"  >
        <img src="images/iiit.png" width="100" height="70" alt="IIIT" class="pngimg" />
	</div>
	
	
			<div id="topMenu">
			<h1 style="text-align: center;">Virtual Lab for IMAGE PROCESSING</h1>
<div class="home">
<a href="index.php">home</a>
</div>

			<div class="menu">


<ul>
<li><a href="objective.php?exp=segment" target="_self" >Objective</a>
</li>
<li><a href="intro.php?exp=segment" target="_self" >Introduction</a>
</li>
<li><a href="theory.php?exp=segment" target="_self" >Theory</a>


</li>
<li><a href="procedure.php?exp=segment" target="_self" >Procedure</a>


</li>
<li><sel><a href="segment.php" target="_self" >Experiment</a></sel>


</li>
<li><a href="#" target="_self" >Assessment</a>
				<ul>
					<li><a href="quiz.php?exp=segment">Quiz</a></li>
					<li><a href="assignment.php?exp=segment">Assignment</a></li>
			   </ul>

</li><!--
<li><a href="references.php?exp=segment" target="_self" >References</a>
</li>
-->

<li><a href="summary.php" target="_blank" >Summary</a>
</li>
</ul>
</div>
			</div>
			
    </div>
	
	<div id="imgInfo" class="reGrow" style="display: none">
	<table>
	<tr> <th width=180>
	Position
	</th> <th></th> </tr>
	<tr class="region" ><td id="st" >x=60 y=60 </td>
	    <th id="mesg" class="ends"></th></tr>
	</table>
	</div>

    <!-- This is 'Parameters_jpg' -->
    <div id="Layer-2">

	  <h2 style="text-align: center;">Image Segmentation</h2>
<div id="nextBox">
</div>
 <div id="navig">
      <?php if (!isset($_GET["assess"]) && $_SESSION["piping"]!=99)  { ?> <button1>Select Image</button1>   <?php } ?>
      <button4>Reset</button4>
      <button2>Run</button2>
	 <?php if($_SESSION["piping"]==99) { ?> <button7>Return to Piping</button7> <?php } ?>
   <br/><br/> </div>


<div class="Parameters">
	
<h3>&nbsp;<input type="radio" name="choice" onclick="
			$('#seedCont').hide(); 
			$('input','#regn').attr('disabled','true');
			$('.region').hide();
			$('input','#thres').removeAttr('disabled');
			
			
			" value="1" checked="true">Histogram Based Segmentation</h3> 
<div class="Parameter" id="thres" style="margin-left: 10px">
<i>Manual Thresholding</i> <br/>
<input type="radio" name="option1" onclick="if(jstate==2) { $('#end').hide();} sing='yes'; " value="Man1">Single Threshold<br>
<input type="radio" name="option1" onclick="if(jstate==2) { $('#end').show();} sing='no'; " value="Man2">Double Threshold<br>
<i>Automatic Threshold</i> <br/>
<input type="radio" name="option1" class="notMan" value="Auto" checked>Automatic (Otsu)<br>
</div>

<h3>&nbsp;<input type="radio" class="notMan" onclick="
		$('input','#thres').attr('disabled','true');
		$('input','#regn').removeAttr('disabled');
		$('#seedCont').show();
		$('#regionG').click();
		
		" name="choice" id= "regionGrowing" value="regn">Region Growing</h3> 
<div class="Parameter" id="regn" style="margin-left: 10px"> 
<input type="radio" name="option2" id="regionG" disabled value="1" onclick="$('#mean').show();$('#variance').hide();">Based on Mean<br>
<div id="mean" class="region" style="margin-left:20px; display:none;">
<b>Range:</b><br/>
<input type="radio" name="option3" value="1">All included Pixels<br>
<input type="radio" name="option3" value="2" checked >Last 10 included Pixels<br>
<b>Deviation:</b>
	&#177;<input type="text" id="slider-Dev-display" maxlength="10" value="10%" size="4"/>
	<div id="slider-Dev" ></div>

	<script type="text/javascript">
	$(function() {
	  $("#slider-Dev").slider({
	  value:10,
	  min: 0,
 	  max: 20,
	  step: 1,
	  slide: function(event, ui) {
	  $("#slider-Dev-display").attr("value",ui.value + "%"); 
	  }
	  });
	});
</script>
</div>

<input type="radio" name="option2" disabled value="2" onclick="$('#variance').show();$('#mean').hide();" >Based on Variance<br>
<div id="variance" class="region" style="margin-left:20px; display:none;">
<b>Range:</b><br/>
<input type="radio" name="option4" value="1">All included Pixels<br>
<input type="radio" name="option4" value="2" checked>Last 10 included Pixels<br>
<b>Deviation:</b>
	&#177;<input type="text" id="slider-Dev2-display" maxlength="10" value="2%" size="4"/>
	<div id="slider-Dev2" ></div>

	<script type="text/javascript">
	$(function() {
	  $("#slider-Dev2").slider({
	  value:2,
	  min: 0,
 	  max: 5,
	  step: 1,
	  slide: function(event, ui) {
	  $("#slider-Dev2-display").attr("value",ui.value + "%"); 
	  }
	  });
	});
</script>
</div>
</div>
</div>

<div id="placeholder" style="margin-left: 20px; width:220px; height: 220px"></div>
</div>
</div>
</body>


