<?php

	session_start();
	$user_folder="images/temp/";	
	//folder where the temporary session images will be saved to
	//you may need to chmod 775 or 777 this folder

$operation=$_GET['opn'];
if($operation=="affine") {

	$inter=$_GET['inter'];	
	$sc1 = $_GET['sc1'];
$sc2 = $_GET['sc2'];
	$ang = $_GET['ang'];
	$xtr = $_GET['xtr'];
	$ytr = $_GET['ytr'];
	$order = $_GET['order'];

	$parm = $inter.' '.$sc1.' '.$sc2.' '.$ang.' '.$xtr.' '.$ytr.' '.$order;	
	$link = 'affine.php?id='.$_SESSION["id"].'&source='.$_SESSION["currentS"].'&inter='.$inter.'&sc1='.$sc1.'&sc2='.$sc2.'&ang='.$ang.'&xtr='.$xtr.'&ytr='.$ytr.'&order='.$order;	




	//echo $parm;

	//Initial Command Change the $parm variable as required now 

	
	$num = (int)exec('./execs/affine.out '.$user_folder.$_SESSION["id"].'_'.$_SESSION["currentS"].'.png '.$user_folder.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png '.$parm);

//                  echo './affine.out '.$user_folder.$_SESSION["id"].'_'.$_SESSION["currentS"].'.png '.$user_folder.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png '.$parm;

	
	//
	//
	$inter_msg="";
	switch($inter) {
	case "1": $inter_msg.="using nearest neighbour interpolation"; break;
	case "2": $inter_msg.="using bilinear interpolation"; break;
	case "3": $inter_msg.="using bicubic interpolation"; break;
	}

	$mesg[0]="";
	$mesg[1]="";
	$mesg[2]="";
	$mesg[3]="";
	$msg="";

	$orr=str_split($order);
	$arr =array($orr[0],$orr[2],$orr[4]) ;
	if($sc1!="0") {
		if($sc2=="0"){
			$mesg[$arr[0]] = "Upscale - ";
		}
		else{
			$mesg[$arr[0]] = "Downscale - ";
		}
		$mesg[$arr[0]].=$sc1."x ".$inter_msg;	
	}
	if($ang!="0") {
		$mesg[$arr[1]].="Rotation - ".$ang." degrees ".$inter_msg;
	}
	if($xtr!="0") {
		$mesg[$arr[2]].="X-translation - ".$xtr." pixels ";
	}
	if($ytr!="0") {
		$mesg[$arr[2]].="Y-translation - ".$ytr." pixels ";
	}
	if($mesg[2]==""){
		$mesg[2]=$mesg[3];
	}

	if($mesg[1]==""){
		$mesg[1]=$mesg[2];
		$mesg[2]=$mesg[3];
	}



	//echo $mesg[$arr[2]];
	//echo $num;

	$exp_no = $_SESSION["state"];
	echo '<div> <div class="item">
		<h2>Initial Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>';

	for($i=1;$i<$num;$i++) {
		echo '<div class="item">
			<h2>Intermediate '.$i.'</h2>
			<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'_'.$i.'.png" alt="picture" width="140px" height="140px"/><br>
  			<dl class="details ui-helper-clearfix"><dt>'.$mesg[$i].'</dt> </dl>
			</div>';
	}
	
	echo '<div class="item">
		<h2>Processed Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/><br>
		<dl class="details ui-helper-clearfix"><dt>'.$mesg[$i].'</dt> </dl>
		</div>';

	echo '</div>';

if($_SESSION["piping"]!=99) {
	$myfile=fopen("temp/".$_SESSION["id"].".html",'a');
	fwrite($myfile,
		'<div> <div class="experiment"><h2 style="display:inline"><u>Experiment '.$exp_no.'</u></h2><a href="'.$link.'"> retry</a><br/><br/><div class="item">
		<h3>Initial Image</h3>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>');

	for($i=1;$i<$num;$i++) {fwrite($myfile,
		'<div class="item">
		<h3>Intermediate '.$i.'</h3>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'_'.$i.'.png" alt="picture" width="140px" height="140px"/><br>
  		<dl class="details ui-helper-clearfix"><dt>'.$mesg[$i].'</dt> </dl>
		</div>');
	}
	fwrite($myfile,'<div class="item">
		<h3>Final Image</h3>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
		</div></div><br/></div>');

	fclose($myfile); }
	if($_SESSION["piping"]!=99) { $_SESSION["state"]++; }
} 
if($operation=="point") {
	$mesg="";
	$opt=$_GET["opt"];
	$link = 'point.php?id='.$_SESSION["id"].'&source='.$_SESSION["currentS"].'&opt='.$opt;
	switch ($opt) {
	case "1": $m=$_GET["m"]; $c=$_GET["c"]; $parm="1 ".$m." ".$c;
		$link.="&m=".$m."&c=".$c;
		$slp = round(tan(3.14 * intval($m)/180),2);
		$mesg.='Linear Transform,<br/>f(r)='.$slp.'r + '.$c;break;

	case "2": $c=$_GET["c"]; $parm="2 ".$c; 
		$link.="&c=".$c;
		$mesg.='Non-Linear Transform,<br/>f(r)='.$c.'log10(1+r)';
		break;
	case "3": $a=$_GET["a"]; $b=$_GET["b"]; $c=$_GET["beta"]; $parm="3 ".$a." ".$b." ".$c; 
		$link.="&a=".$a."&b=".$b."&beta=".$c;
		$mesg='Clipping,<br/>between '.$a.' and '.$b.',slope='.$c;
		break;
	case "4": $a=$_GET["a"]; $b=$_GET["b"]; $c=$_GET["beta"]; $parm="4 ".$a." ".$b." ".$c;
		$link.="&a=".$a."&b=".$b."&beta=".$c;
		$mesg='Windowing,<br/>between '.$a.' and '.$b.',slope='.$c;
	       	break;
	}
//	echo $mesg;
	exec ("./execs/point.out ".$user_folder.$_SESSION["id"].'_'.$_SESSION["currentS"].'.png '.$user_folder.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png '.$parm);
	//echo "./point.out ".$user_folder.$_SESSION["id"].'_'.$_SESSION["currentS"].'.png '.$user_folder.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png '.$parm;

	$exp_no = $_SESSION["state"];
	
	if($_SESSION["piping"]!=99) {
$myfile=fopen("temp/".$_SESSION["id"].".html",'a');
	fwrite($myfile,
		'<div> <div class="experiment"><h2 style="display:inline"><u>Experiment '.$exp_no.'</u></h2><a href="'.$link.'"> retry</a><br/><br/>
		<div class="item">
		<h3>Initial Image</h3>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>
		<div class="item">
		<h3>Processed Image</h3>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>'.$mesg.'</dt> </dl>
		</div><br><br></div></div>');
	fclose($myfile); 
	}


	echo '<div class="item">
		<h2>Initial Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>';
	echo ' <div class="item">
		<h2>Processed Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>'.$mesg.'</dt> </dl>
		</div>';
	if($_SESSION["piping"]!=99) { $_SESSION["state"]++; }	

}
if($operation=="arith") {
	$opt=$_GET["opt"];
	$fitting=$_GET["fitting"];
	$image=$_GET["image"];

	exec("./execs/arith.out ".$user_folder.$_SESSION["id"].'_'.$_SESSION["currentS"].'.png arith/'.$image.' '.$user_folder.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png '.$opt.' '.$fitting);
//echo "./arith.out ".$user_folder.$_SESSION["id"].'_'.$_SESSION["currentS"].'.png arith/'.$image.' '.$user_folder.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png '.$opt.' '.$fitting;
	$link = 'arith.php?id='.$_SESSION["id"].'&source='.$_SESSION["currentS"].'&opt='.$opt.'&fit='.$fitting.'&image='.$image;
	
	$opern = "";

	switch ($opt) {
	case "1": $opern = 'Addition and ';break;
	case "2": $opern = 'Subtraction and ';break;
	case "3": $opern = 'Difference and ';break;
	case "4": $opern = 'Multiplication and ';break;
	case "5": $opern = 'Division and ';break;
	}

	$map_fn="";

	switch ($fitting){
		case "1": $map_fn = 'Clipping';break;
		case "2": $map_fn = 'Auto-Scaling';break;
	}
	$exp_no = $_SESSION["state"];
	if($_SESSION["piping"]!=99) { $_SESSION["state"]++; }

	echo '<div> <div class="item">
		<h2>Primary Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>
		<div class="item">
		<h2>Secondary Image</h2>
		<img src="arith/'.$image.'" alt="picture" width="140px" height="140px"/>
		</div><div class="item">
		<h2>Processed Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>'.$opern.$map_fn.'</dt> </dl>
		</div><br><br></div>';

		if($_SESSION["piping"]!=99) {
		
	$myfile=fopen("temp/".$_SESSION["id"].".html",'a');
	fwrite($myfile,'
		<div> <div class="experiment">
		<h2><u>Experiment '.$exp_no.'</u></h2>
		<div class="item">
		<h2>Primary Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>
		<div class="item">
		<h2>Secondary Image</h2>
		<img src="arith/'.$image.'" alt="picture" width="140px" height="140px"/>
		</div><div class="item">
		<h2>Processed Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>'.$opern.$map_fn.'</dt> </dl>
		</div><br><br></div></div>');
	fclose($myfile); 
	}
}
if($operation=="histo") {

	
	$mode=$_GET["mode"];

if($mode!="point" && $mode!="segment") {
	
	$link='';
	if($mode==1||$mode==2) {
	$bins=$_GET["bins"];
	$parm=$bins;
	$link=$link.'&bins='.$bins;
	}
	if($mode==4) {
	$link=$link.'&wsize='.$wsize;
	$parm=$_GET["wsize"]." ".$_GET["wsize"];
	}
	if($mode==5) {
	$size=$_GET["size"];
	$link=$link.'&size='.$size;
	$parm=$size;
	}
	$out=exec("./execs/hist.out ".$user_folder.$_SESSION["id"].'_'.$_SESSION["currentS"].'.png '.$user_folder.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png '.$mode.' '.$parm);
	
	$link = 'histo.php?id='.$_SESSION["id"].'&source='.$_SESSION["currentS"].'&mode='.$mode.$link;
	
		
	$exp_no = $_SESSION["state"];
	

$props=explode(" ",$out);	
if($mode==1) {

	echo '<div> <div class="item">
		<h2>Source Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>
		<div class="item">
		<h2>Histogram</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt> using '.$bins.' bins.</dt> <dt>Min='.$props[0].'&nbsp;&nbsp;Avg='.$props[2].'&nbsp;&nbsp;Max='.$props[1].'</dt></dl>
		</div><br><br></div>';

		if($_SESSION["piping"]!=99) {
	$myfile=fopen("temp/".$_SESSION["id"].".html",'a');
	fwrite($myfile,'
		<div> <div class="experiment">
		<h2 style="display:inline"><u>Experiment '.$exp_no.'</u></h2><a href="'.$link.'"> retry</a><br/><br/><div class="item">
				<h2>Source Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>
		<div class="item">
		<h2>Histogram</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt> using '.$bins.' bins.</dt> <dt>Min='.$props[0].'&nbsp;&nbsp;Avg='.$props[2].'&nbsp;&nbsp;Max='.$props[1].'</dt></dl>
		</div><br><br></div></div>');
	fclose($myfile); 
	}
} 
if($mode==2) {
echo '<div><div class="item">
		<h2>Source Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>';
  for($i=1;$i<=4;$i++) {
	echo '
		<div class="item">
		<h2>Subhistogram '.$i.'</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).$i.'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt> using '.$bins.' bins.</dt> <dt>Min='.$props[3*$i-3].'&nbsp;&nbsp;Avg='.$props[3*$i-1].'&nbsp;&nbsp;Max='.$props[$i*3-2].'</dt></dl>
		</div>';
		}
echo '</div>';
if($_SESSION["piping"]!=99) {
	$myfile=fopen("temp/".$_SESSION["id"].".html",'a');
	fwrite($myfile,'
		<div> <div class="experiment">
		<h2 style="display:inline"><u>Experiment '.$exp_no.'</u></h2><a href="'.$link.'"> retry</a><br/><br/><div class="item">
		<h2>Source Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>'); 
 for($i=1;$i<=4;$i++) {
 fwrite($myfile,'
		<div class="item">
		<h2>Subhistogram '.$i.'</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).$i.'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt> using '.$bins.' bins.</dt> <dt name="props">Min='.$props[3*$i-3].'&nbsp;&nbsp;Avg='.$props[3*$i-1].'&nbsp;&nbsp;Max='.$props[$i*3-2].'</dt></dl>
		</div><br><br>');

	}
	fwrite($myfile,'</div>');	
	fclose($myfile); 
}
	}
if($mode==3 || $mode==4 || $mode==5) {
switch ($mode) {
	case 3 : $msg="using Global Histogram Equalization."; break;
	case 4 : $msg="using Adaptive Histogram Equilization."; break;
	case 5 : $msg = "after Filtering of Histogram using filter of size ".$size; break;
	}
	
		echo '<div> <div class="item">
		<h2>Source Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>
		<div class="item">
		<h2>Processed Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>'.$msg.'</dt></dl>
		</div><br><br></div>';

	if($_SESSION["piping"]!=99) {
	$myfile=fopen("temp/".$_SESSION["id"].".html",'a');
	fwrite($myfile,'
		<div> <div class="experiment">
		<h2 style="display:inline"><u>Experiment '.$exp_no.'</u></h2><a href="'.$link.'"> retry</a><br/><br/><div class="item">
				<h2>Source Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>
		<div class="item">
		<h2>Histogram</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>'.$msg.'</dt></dl>
		</div><br><br></div></div>');
	fclose($myfile); }
}

} else { 

if($mode=="point") {
	$out=exec('./execs/hist.out '.$user_folder.$_SESSION["id"].'_'.$_SESSION["currentS"].'.png '.$user_folder.$_SESSION["id"].'_'.$_SESSION["currentS"].'_1.png '.$bins.' 1');
$props=explode(" ",$out);	
	echo '<div class="item">
		<h2>Source Histogram</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'_1.png" alt="picture" width="140px" height="140px"/>
<dl class="details ui-helper-clearfix"><dt> using '.$bins.' bins.</dt> <dt>Min='.$props[0].'&nbsp;&nbsp;Avg='.$props[2].'&nbsp;&nbsp;Max='.$props[1].'</dt></dl>
		</div>';
	if($_SESSION["piping"]!=99) {
	
	$myfile=fopen("temp/".$_SESSION["id"].".html",'a');
	fwrite($myfile,'<div> <div class="experiment">
		<h3><u>Histograms for Above Experiment</u></h3>
<div class="item">
		<h2>Source Histogram</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'_1.png" alt="picture" width="140px" height="140px"/>
<dl class="details ui-helper-clearfix"><dt> using '.$bins.' bins.</dt> <dt>Min='.$props[0].'&nbsp;&nbsp;Avg='.$props[2].'&nbsp;&nbsp;Max='.$props[1].'</dt></dl>
		</div>');
	}


	$out=exec('./execs/hist.out '.$user_folder.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png '.$user_folder.$_SESSION["id"].'_'.($_SESSION["state"]+1).'_1.png '.$bins.' 1');
$props=explode(" ",$out);	
echo '<div class="item">
		<h2>Output Histogram</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'_1.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt> using '.$bins.' bins.</dt> <dt>Min='.$props[1].'&nbsp;&nbsp;Avg='.$props[3].'&nbsp;&nbsp;Max='.$props[2].'</dt></dl>
		</div>';

if($_SESSION["piping"]!=99) {
	
	fwrite($myfile,'
		<div class="item">
		<h2>Output Histogram</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'_1.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt> using '.$bins.' bins.</dt> <dt>Min='.$props[1].'&nbsp;&nbsp;Avg='.$props[3].'&nbsp;&nbsp;Max='.$props[2].'</dt></dl>
		</div><br><br></div></div>');
	fclose($myfile); 
	}
} 
if($mode=="segment") {
	$out=exec('./execs/hist.out '.$user_folder.$_SESSION["id"].'_'.$_SESSION["currentS"].'.png '.$user_folder.$_SESSION["id"].'_'.$_SESSION["currentS"].'_1.png '.$bins.' 1');
$props=explode(" ",$out);	
	echo '<h2>Histogram of Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'_1.png" alt="picture" width="140px" height="140px"/>
         Min='.$props[0].'&nbsp;&nbsp;Avg='.$props[2].'&nbsp;&nbsp;Max='.$props[1].'
		</div>';
	
	if($_SESSION["piping"]!=99) {
	
	$myfile=fopen("temp/".$_SESSION["id"].".html",'a');
	fwrite($myfile,'<div> <div class="experiment">
		<h2><u>Experiment '.$exp_no.'</u></h2>
		<div class="item">
		<h2>Source Histogram</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'_1.png" alt="picture" width="140px" height="140px"/>
<dl class="details ui-helper-clearfix"><dt> using '.$bins.' bins.</dt> <dt>Min='.$props[0].'&nbsp;&nbsp;Avg='.$props[2].'&nbsp;&nbsp;Max='.$props[1].'</dt></dl>
		</div>');
}
}


}

if($_SESSION["piping"]!=99) { $_SESSION["state"]++; }

}
if($operation=="diff") {

$exp=$_GET["exp"];

$link = 'diff.php?id='.$_SESSION["id"].'&source='.$_SESSION["currentS"].'&exp='.$exp;

switch ($exp) {
case 1: 
 $type=$_GET["type"];
 $metric=$_GET["metric"];
 $sx=$_GET["sx"];
 $sy=$_GET["sy"];
 $link.="&type=".$type."&metric=".$metric."&sx=".$sx."&sy=".$sy;
 switch($type) {
 case 1:
 $ex=$_GET["ex"];
 $ey=$_GET["ey"]; 
 $link.="&ex=".$ex."&ey=".$ey;
 $out=exec("./execs/dist.out 1 ".$sx." ".$sy." ".$ex." ".$ey." ".$metric);
 echo $out;
  break;
 case 2:
  $dist=$_GET["dist"];
  $link.="&dist=".$dist;
 $out=exec("./execs/dist.out 2 images/blank.png ".$user_folder.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png '.$sy." ".$sx." ".$dist." ".$metric);

 $exp_no = $_SESSION["state"] + 1;

 if($_SESSION["piping"]!=99) {
 $myfile=fopen("temp/".$_SESSION["id"].".html",'a');
	fwrite($myfile,
		'<div class="experiment"><h2 style="display:inline"><u>Experiment '.$exp_no.'</u></h2><a href="'.$link.'"> retry</a><br/><br/>
		<div class="item">
		<h3>Processed Image</h3>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>'.$out.'</dt> </dl>
		</div><br><br></div>');
	fclose($myfile); 
}

	echo '<div> ';
	echo ' <div class="item">
		<h2>Processed Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>'.$out.'</dt> </dl>
		</div><br><br></div>';
	if($_SESSION["piping"]!=99) { $_SESSION["state"]++; }
 break;
 case 3:
 $out=exec("./execs/dist.out 3 images/BW_liz.bmp ".$user_folder.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png '.$sy." ".$sx." ".$metric);
 
 $exp_no = $_SESSION["state"] + 1;
 
 if($_SESSION["piping"]!=99) {
	
 $myfile=fopen("temp/".$_SESSION["id"].".html",'a');
	fwrite($myfile,
		'<div class="experiment"><h2 style="display:inline"><u>Experiment '.$exp_no.'</u></h2><a href="'.$link.'"> retry</a><br/><br/>
		<div class="item">
		<h3>Initial Image</h3>
		<img src="images/BW_liz.bmp" alt="picture" width="140px" height="140px"/>
		</div>
		<div class="item">
		<h3>Processed Image</h3>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>'.$out.'</dt> </dl>
		</div><br><br></div>');
	fclose($myfile); 
}

	echo '<div> <div class="item">

		<h2>Initial Image</h2>
		<img src="images/BW_liz.bmp" alt="picture" width="140px" height="140px"/>
		</div>';
	echo ' <div class="item">
		<h2>Processed Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>'.$out.'</dt> </dl>
		</div><br><br></div>';
	if($_SESSION["piping"]!=99) { $_SESSION["state"]++; }
 break;
}

break;

case 2:
 $type=$_GET["type"];
 $sx=$_GET["sx"];
 $sy=$_GET["sy"];
 $ex=$_GET["ex"];
 $ey=$_GET["ey"]; 
 $out=exec("./execs/path.out ".$user_folder.$_SESSION["id"].'_'.$_SESSION["currentS"].'.png '.$user_folder.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png '.$sy." ".$sx." ".$ey." ".$ex." ".$type);
 $link.="&type=".$type."&sx=".$sx."&sy=".$sy."&ex=".$ex."&ey=".$ey;
 
 if($out!="0") {
 
	$exp_no = $_SESSION["state"];
	if($_SESSION["piping"]!=99) { $_SESSION["state"]++; }

if($_SESSION["piping"]!=99) {
	
$myfile=fopen("temp/".$_SESSION["id"].".html",'a');
	fwrite($myfile,
		'<div class="experiment"><h2 style="display:inline"><u>Experiment '.$exp_no.'</u></h2><a href="'.$link.'"> retry</a><br/><br/>
		<div class="item">
		<h3>Initial Image</h3>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>
		<div class="item">
		<h3>Processed Image</h3>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>'.$out.'</dt> </dl>
		</div><br><br></div>');
	fclose($myfile); 
}

	echo '<div> <div class="item">

		<h2>Initial Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>';
	echo ' <div class="item">
		<h2>Processed Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>'.$out.'</dt> </dl>
		</div><br><br></div>';
}else {
  echo '0';
}
break;

}

 
}
if($operation=="neigh") {
if($_GET["args"]!="6") {

 $temp=explode("_",$_GET["args"]);
 $link = 'neigh.php?id='.$_SESSION["id"].'&source='.$_SESSION["currentS"].'&type='.($temp[0]+1).'&size='.$temp[1];	


 switch ($temp[0]) {
	case 1: $msg="Average filter of size ".$temp[1]."x".$temp[1].".";
	break;
	case 2: $msg="Triangular filter of size ".$temp[1]."x".$temp[1].".";
	break;
	case 3: $msg="Gaussian filter of size ".$temp[1]."x".$temp[1].".";
	break;
	case 4: $msg="Custom filter of size ".$temp[1]."x".$temp[1].".";
	break;
	case 5: $msg="Median filtering with window of size ".$temp[2]." x".$temp[2].".";
	break;
	}
	
 
 $args=$temp[1].$temp[2];
} else {
	$args=$_GET["args"];
	$msg="Unsharp Masking.";
	}

exec("./execs/neigh.out ".$user_folder.$_SESSION["id"].'_'.$_SESSION["currentS"].'.png '.$user_folder.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png '.$args);

$exp_no = $_SESSION["state"];

if($_SESSION["piping"]!=99) {
	
$myfile=fopen("temp/".$_SESSION["id"].".html",'a');
	fwrite($myfile,
		'<div class="experiment"><h2 style="display:inline"><u>Experiment '.$exp_no.'</u></h2><a href="'.$link.'"> retry</a><br/><br/>
		<div class="item">
		<h3>Initial Image</h3>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>
		<div class="item">
		<h3>Processed Image</h3>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>Image Processed using '.$msg.'</dt> </dl>
		</div><br><br></div>');
	fclose($myfile); 
}

	echo '<div> <div class="item">

		<h2>Initial Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>';
	echo ' <div class="item">
		<h2>Processed Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>Image Processed using '.$msg.'</dt> </dl>
		</div><br><br></div>';
if($_SESSION["piping"]!=99) { $_SESSION["state"]++; }
 
} 
if($operation=="colour") {

	$mesg="";
	$opt=$_GET["args"];
	$link = 'colour.php?id='.$_SESSION["id"].'&source='.$_SESSION["currentS"];
	$expno = $_SESSION["state"]; 
		
	switch ($opt) {
	case "1": 
		$parm=$_GET["spc"];
		$link=$link."&mode=1&space=".$parm;
		$out=exec ('./execs/colour.out '.$user_folder.$_SESSION["id"].''.$_SESSION["currentS"].'.jpg '.$user_folder.$_SESSION["id"].($_SESSION["state"]+1).'.jpg '.$parm);
		
		echo $out.' 	  <table style="text-align:center;"> <tr> <td>
	<img src="images/temp/'.$_SESSION["id"].($_SESSION["state"]+1).'1.jpg"  alt="InputLayer" width="130px" style="margin: 0 20px 0 20px;" height="130px"/> </td><td>
	<img src="images/temp/'.$_SESSION["id"].($_SESSION["state"]+1).'2.jpg"  alt="InputLayer" width="130px" style="margin: 0 20px 0 20px;" height="130px"/> </td><td>
	<img src="images/temp/'.$_SESSION["id"].($_SESSION["state"]+1).'3.jpg"  alt="InputLayer" width="130px" style="margin: 0 20px 0 20px;" height="130px"/> </td></tr><tr>
	';

if($_SESSION["piping"]!=99) {
	$myfile=fopen("temp/".$_SESSION["id"].".html",'a');
	fwrite($myfile,
		'<div class="experiment"><h2 style="display:inline"><u>Experiment '.$expno.'</u></h2><a href="'.$link.'"> retry</a><br/><br/>
		<table style="text-align:center;"> <tr> <td>
		<img src="images/temp/'.$_SESSION["id"].''.($_SESSION["currentS"]).'.jpg"  alt="InputLayer" width="130px" style="margin: 0 20px 0 20px;" height="130px"/> </td><td>
	<img src="images/temp/'.$_SESSION["id"].($_SESSION["state"]+1).'1.jpg"  alt="InputLayer" width="130px" style="margin: 0 20px 0 20px;" height="130px"/> </td><td>
	<img src="images/temp/'.$_SESSION["id"].($_SESSION["state"]+1).'2.jpg"  alt="InputLayer" width="130px" style="margin: 0 20px 0 20px;" height="130px"/> </td><td>
	<img src="images/temp/'.$_SESSION["id"].($_SESSION["state"]+1).'3.jpg"  alt="InputLayer" width="130px" style="margin: 0 20px 0 20px;" height="130px"/> </td></tr><tr>
	<td>Input Image</td>
	');
}
	
	switch ($parm) {
	case "1" :
	echo '<td>Hue</td><td>Saturation</td><td>Intensity</td></tr></table>';	
if($_SESSION["piping"]!=99) {
		fwrite($myfile,'<td>Hue</td><td>Saturation</td><td>Intensity</td></tr></table>'); }
	break;
	case "2" :
	echo '<td>Cyan</td><td>Magenta</td><td>Yellow</td></tr></table>';	
	if($_SESSION["piping"]!=99) {
	fwrite($myfile,'<td>Cyan</td><td>Magenta</td><td>Yellow</td></tr></table>');	 }
	break;
	case "3" :
	echo '<td>Luminance</td><td>Blue-Chroma</td><td>Red-Chroma</td></tr></table>';	
	if($_SESSION["piping"]!=99) {
	fwrite($myfile,'<td>Luminance</td><td>Blue-Chroma</td><td>Red-Chroma</td></tr></table>');	 }
	break;
	}
	if($_SESSION["piping"]!=99) {
	
	fwrite($myfile,"</div>");
	fclose($myfile); }
		break;
		
		
	default:
	$space=$_GET["space"];
		switch($space) {
			case "1" : $msg='to the RGB space';
					break;
			case "2" : $msg='to the HSI space'; 
					break;
			case "3" : $msg= 'to the CMY space';
					break;
		}
		$opt2=$_GET["opt"];
		$f=$_GET["first"];
		$s=$_GET["second"];
		$t=$_GET["third"];

		$plane=($f*1)+($s*2)+($t*4);
				$link=$link."&mode=2&space=".$space."&plane=".$plane;
		if($opt2=="lin") {
		
			$m=$_GET["m"]; $c=$_GET["c"]; 
			$link=$link."&mode2=1&m=".$m."&c=".$c;
			$msg =' Linear Transform,<br/>f(r)='.$m.'r + '.$c .' '.$msg;
			$parm = $plane." ".$m." ".$c." ".$space;
			$slp = round(tan(3.14 * intval($m)/180),2);
			$mesg.='Linear Transform,<br/>f(r)='.$slp.'r + '.$c;
			exec ("./execs/colour2.out ".$user_folder.$_SESSION["id"].''.$_SESSION["currentS"].'.jpg '.$user_folder.$_SESSION["id"].''.($_SESSION["state"]+1).'.jpg '.$parm);
			echo '<div> <div class="item">

		<h2>Initial Image</h2>
		<img src="images/temp/'.$_SESSION["id"].''.($_SESSION["currentS"]).'.jpg" alt="picture" width="140px" height="140px"/>
		</div>';
		
		
	echo ' <div class="item">
		<h2>Processed Image</h2>
		<img src="images/temp/'.$_SESSION["id"].''.($_SESSION["state"]+1).'.jpg" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>Image Processed applying '.$msg.'</dt> </dl>
		</div><br><br></div>';
   
	
	if($_SESSION["piping"]!=99) {
	
			$myfile=fopen("temp/".$_SESSION["id"].".html",'a');
	fwrite($myfile,
		'<div class="experiment"><h2 style="display:inline"><u>Experiment '.$expno.'</u></h2><a href="'.$link.'"> retry</a><br/><br/>
		<div class="item">
		<h2>Initial Image</h2>
		<img src="images/temp/'.$_SESSION["id"].''.($_SESSION["currentS"]).'.jpg" alt="picture" width="140px" height="140px"/>
		</div>
		<div class="item">
		<h2>Processed Image</h2>
		<img src="images/temp/'.$_SESSION["id"].''.($_SESSION["state"]+1).'.jpg" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>Image Processed applying '.$msg.'</dt> </dl>
		</div><br><br></div>');
	fclose($myfile); 
	}
	
			} else {
			$link=$link."&mode2=2&opt2=".$opt2;
			if($opt2=="1") {
			$msg = ' Global Histogram Equilization '.$msg;
			} else {
			$msg = ' Adaptive Histogram Equilization '.$msg;
			$wsize=$_GET["wsize"]." ".$_GET["wsize"];
			$link=$link."&wsize=".$_GET["wsize"];
			}
			
			exec ("./execs/colour3.out ".$user_folder.$_SESSION["id"].''.$_SESSION["currentS"].'.jpg '.$user_folder.$_SESSION["id"].''.($_SESSION["state"]+1).'.jpg '.$opt2.' '.$space.' '.$plane.' '.$wsize);
				echo '<div> <div class="item">

		<h2>Initial Image</h2>
		<img src="images/temp/'.$_SESSION["id"].''.($_SESSION["currentS"]).'.jpg" alt="picture" width="140px" height="140px"/>
		</div>';
	echo ' <div class="item">
		<h2>Processed Image</h2>
		<img src="images/temp/'.$_SESSION["id"].''.($_SESSION["state"]+1).'.jpg" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>Image Processed applying '.$msg.'</dt> </dl>
		</div><br><br></div>';
		
	if($_SESSION["piping"]!=99) {
		
					$myfile=fopen("temp/".$_SESSION["id"].".html",'a');
	fwrite($myfile,
		'<div class="experiment"><h2 style="display:inline"><u>Experiment '.$expno.'</u></h2><a href="'.$link.'"> retry</a><br/><br/>
		<div> <div class="item">

		<h2>Initial Image</h2>
		<img src="images/temp/'.$_SESSION["id"].''.($_SESSION["currentS"]).'.jpg" alt="picture" width="140px" height="140px"/>
		</div>
		<div class="item">
		<h2>Processed Image</h2>
		<img src="images/temp/'.$_SESSION["id"].''.($_SESSION["state"]+1).'.jpg" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>Image Processed applying '.$msg.'</dt> </dl>
		</div><br><br></div');
	fclose($myfile); 
	}	
			}
			

		}	

	$expno = $_SESSION["state"];
	if($_SESSION["piping"]!=99) { $_SESSION["state"]++; }	


}

if($operation=="morph") {
	$otype=$_GET["otype"];
	$stype=$_GET["stype"];
	$size=$_GET["size"];
	$ang=$_GET["ang"];

	switch ($otype) {
	 case '2': $msg="Erosion"; break;
	 case '1': $msg="Dilation"; break;
	 case '4': $msg="Opening"; break;
	 case '3': $msg="Closing"; break;
	 }
	
	exec("./execs/morpho.out ".$user_folder.$_SESSION["id"].'_'.$_SESSION["currentS"].'.png '.$user_folder.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png '.$otype.' '.$stype.' '.$size);
	$link = 'morph.php?id='.$_SESSION["id"].'&source='.$_SESSION["currentS"].'&opt='.$opt.'&fit='.$fitting.'&image='.$image;
	
	$exp_no = $_SESSION["state"];

if($_SESSION["piping"]!=99) {
	
	$myfile=fopen("temp/".$_SESSION["id"].".html",'a');
	fwrite($myfile,
		'<div class="experiment"><h2 style="display:inline"><u>Experiment '.$exp_no.'</u></h2><a href="'.$link.'"> retry</a><br/><br/>
		<div class="item">
		<h3>Initial Image</h3>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>
		<div class="item">
		<h3>Processed Image</h3>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>'.$msg.'Performed on Input Image.</dt> </dl>
		</div><br><br></div>');
	fclose($myfile); 
}

	echo '<div> <div class="item">

		<h2>Initial Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>';
	echo ' <div class="item">
		<h2>Processed Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>Image Processed using '.$msg.'</dt> </dl>
		</div><br><br></div>';
	if($_SESSION["piping"]!=99) { $_SESSION["state"]++; }
}


if($operation=="fourier") {
	$opt=$_GET["opt"];
	if($opt==1) {
			exec("./execs/fourier.out ".$opt.' '.$user_folder.$_SESSION["id"].'_'.$_SESSION["currentS"].'.png '.$user_folder.$_SESSION["id"].'_'.($_SESSION["state"]+1).'1.png '.$user_folder.$_SESSION["id"].'_'.($_SESSION["state"]+1).'2.png');
			$link = 'fourier.php?id='.$_SESSION["id"].'&source='.$_SESSION["currentS"].'&opt='.$opt;
	
			echo '<div> <div class="item">
			<h2>Initial Image</h2>
			<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
			</div>';
			echo ' <div class="item">
			<h2>Amplitude </h2>
			<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'1.png" alt="picture" width="140px" height="140px"/>
			<dl class="details ui-helper-clearfix"><dt>Amplitude Image of the Fourier Transform.</dt> </dl>
			</div>
			<div class="item">
			<h2>Phase</h2>
			<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'2.png" alt="picture" width="140px" height="140px"/>
			<dl class="details ui-helper-clearfix"><dt>Phase Image of the Fourier Transform.</dt> </dl>
			</div>
			
			<br><br></div>';
	} else {
		if($opt==2) {
			$args="./fourier/choice".$_GET["img"].".png";
			$link = 'fourier.php?id='.$_SESSION["id"].'&source='.$_SESSION["currentS"].'&opt='.$opt.'&img='.$_GET["img"];
	
			exec("./execs/fourier.out ".$opt.' '.$args.' '.$user_folder.$_SESSION["id"].'_'.$_SESSION["currentS"].'.png '.$user_folder.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png');
			$link = 'arith.php?id='.$_SESSION["id"].'&source='.$_SESSION["currentS"].'&opt='.$opt.'&fit='.$fitting.'&image='.$image;
	
			$exp_no = $_SESSION["state"];

			if($_SESSION["piping"]!=99) {

			$myfile=fopen("temp/".$_SESSION["id"].".html",'a');
			fwrite($myfile,
				'<div class="experiment"><h2 style="display:inline"><u>Experiment '.$exp_no.'</u></h2><a href="'.$link.'"> retry</a><br/><br/>
				<div class="item">
				<h3>Initial Image</h3>
				<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
				</div>
				<div class="item">
				<h3>Processed Image</h3>
				<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
				<dl class="details ui-helper-clearfix"><dt>'.$msg.'Performed on Input Image.</dt> </dl>
				</div><br><br></div>');
			fclose($myfile); 
			}

			echo '<div> <div class="item">

				<h2>Initial Image</h2>
				<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
				</div>';
			echo ' <div class="item">
				<h2>Processed Image</h2>
				<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
				<dl class="details ui-helper-clearfix"><dt>Image Processed using '.$msg.'</dt> </dl>
				</div><br><br></div>';
		} else {			
			$th=$_GET["th"]; $r=$_GET["r"];
			$dth=$_GET["dth"]; $dr=$_GET["dr"];
			$args=$th.' '.$r.' '.$dth.' '.$dr;
			exec("./execs/fourier.out ".$opt.' '.$user_folder.$_SESSION["id"].'_'.$_SESSION["currentS"].'.png '.$user_folder.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png '.$args);
			
			$link = 'fourier.php?id='.$_SESSION["id"].'&source='.$_SESSION["currentS"].'&opt='.$opt.'&th='.$th.'&r='.$r.'&dth='.$dth.'&dr='.$dr;
			
			$exp_no = $_SESSION["state"];

			if($_SESSION["piping"]!=99) {

			$myfile=fopen("temp/".$_SESSION["id"].".html",'a');
			fwrite($myfile,
				'<div class="experiment"><h2 style="display:inline"><u>Experiment '.$exp_no.'</u></h2><a href="'.$link.'"> retry</a><br/><br/>
				<div class="item">
				<h3>Initial Image</h3>
				<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
				</div>
				<div class="item">
				<h3>Processed Image</h3>
				<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
				<dl class="details ui-helper-clearfix"><dt>'.$msg.'Performed on Input Image.</dt> </dl>
				</div><br><br></div>');
			fclose($myfile); 
			}

			echo '<div> <div class="item">

				<h2>Initial Image</h2>
				<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
				</div>';
			echo ' <div class="item">
				<h2>Processed Image</h2>
				<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
				<dl class="details ui-helper-clearfix"><dt>Image Processed using '.$msg.'</dt> </dl>
				</div><br><br></div>';
		}	
	}
	if($_SESSION["piping"]!=99) { $_SESSION["state"]++; }
}

if($operation=="segment") {
	$mode=$_GET["args"];
	$link = 'segment.php?id='.$_SESSION["id"].'&source='.$_SESSION["currentS"].'&args='.$args;
	switch ($mode) {
		case "Man1":
			$link.="&startx=".$_GET["startx"];
			$args="1 1 ".($_GET["startx"]*4);
			$msg = "Single Threshold segmentation";
			break;
		case "Man2":
			$link.="&startx=".$_GET["startx"];
			$link.="&endx=".$_GET["endx"];
			$args="1 2 ".($_GET["startx"]*4)." ".($_GET["endx"]*4);
			$msg = "Double Threshold segmentation";
			break;
		case "Auto":
			$args="2";
			$msg = "Otzu Thresholding";
			break;
		case "regn":
		$link.="&option2=".$_GET["option2"].'&x='.$_GET["x"].'&y='.$_GET['y'].'&option3='.$_GET["option3"].'&dev='.$_GET["dev"];
		$args="3 ".$_GET["option2"].' '.$_GET["x"].' '.$_GET['y'].' '.$_GET["option3"].' '.$_GET["dev"];
		$msg = "Region growing";
		break;	
	}

	$out=exec('./execs/segment.out '.$user_folder.$_SESSION["id"].'_'.$_SESSION["currentS"].'.png '.$user_folder.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png '.$args);
	
	$exp_no = $_SESSION["state"];

if($_SESSION["piping"]!=99) {
	
	$myfile=fopen("temp/".$_SESSION["id"].".html",'a');
	fwrite($myfile,
		'<div class="experiment"><h2 style="display:inline"><u>Experiment '.$exp_no.'</u></h2><a href="'.$link.'"> retry</a><br/><br/>
		<div class="item">
		<h3>Initial Image</h3>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>
		<div class="item">
		<h3>Processed Image</h3>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>'.$msg.' performed on Input Image.</dt> </dl>
		</div><br><br></div>');
	fclose($myfile); 
}

	echo '<div> <div class="item">

		<h2>Initial Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["currentS"]).'.png" alt="picture" width="140px" height="140px"/>
		</div>';
	echo ' <div class="item">
		<h2>Processed Image</h2>
		<img src="images/temp/'.$_SESSION["id"].'_'.($_SESSION["state"]+1).'.png" alt="picture" width="140px" height="140px"/>
  		<dl class="details ui-helper-clearfix"><dt>'.$msg.' performed on Input Image.</dt> </dl>
		</div><br><br></div>';
	if($_SESSION["piping"]!=99) { $_SESSION["state"]++; }
}


?>

