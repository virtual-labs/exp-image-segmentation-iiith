<?php

session_start();
$_SESSION["state"]++;

$exp='';
$ext='.png';
if(isset($_POST['exp'])) {
$exp=$_POST['exp'];
}
if($exp=='col') {
$ext='.jpg';
}
$user_folder = 'images/temp/';
require('inc/imagemanipulation.php');
$objImage = new ImageManipulation('images/Mosaic'.$exp.$ext);

if ( $objImage->imageok ) {
  $objImage->setCrop($_POST['x'], $_POST['y'], $_POST['w'], $_POST['h']);
  if($exp=='col') {
  $objImage->save($user_folder.$_SESSION["id"].$_SESSION["state"].$ext);
  }
  else {
  $objImage->save($user_folder.$_SESSION["id"].'_'.$_SESSION["state"].$ext);
  
  }
} else {
  echo 'Error!';
}

$_SESSION["currentS"]=$_SESSION["state"];
if($exp!='col') {
if($_SESSION["piping"]==99) {
$text='
<div class="item">
	      <h2>Cropped Image</h2>
	      <img src="images/temp/'.$_SESSION["id"].'_'.$_SESSION["state"].$ext.'" alt="picture" width="281px" height="281px"/>
	      <dl class="details ui-helper-clearfix">
			<dt>
			This is the cropped image.</dt>
			<dt>
			
			Proceed by selecting the operation you want to perform.
			
			</dt>
	      </dl>
	    </div>';
		echo $text;
			
		writethis($text);
} else {
echo '<img
src="images/temp/'.$_SESSION["id"].'_'.$_SESSION["state"].$ext.'" alt="picture" width="281px" height="281px"/>';
}
} else {
exec ("./execs/colour.out ".$user_folder.$_SESSION["id"].$_SESSION["state"].$ext.' '.$user_folder.$_SESSION["id"].($_SESSION["state"]).'s'.$ext.' 4');
echo'

<table style="text-align:center;"> <tr> <td>
 <img src="images/temp/'.$_SESSION["id"].$_SESSION["state"].$ext.'" id="both" alt="InputLayer" width="130px" height="130px"/> </td><td>
	<img src="images/temp/'.$_SESSION["id"].($_SESSION["state"]).'s1'.$ext.'" class="rgb" alt="InputLayer" width="130px"  height="130px"/> </td><td>
	<img src="images/temp/'.$_SESSION["id"].($_SESSION["state"]).'s2'.$ext.'" class="rgb" alt="InputLayer" width="130px"  height="130px"/> </td><td>
	<img src="images/temp/'.$_SESSION["id"].($_SESSION["state"]).'s3'.$ext.'" class="rgb" alt="InputLayer" width="130px"  height="130px"/> </td></tr><tr>
	<td>Original</td><td>Red Plane</td><td>Green Plane</td><td>Blue Plane</td></tr></table> ';
}

function writethis($item)
{$item=preg_replace( '/140/','281', $item);
$myfile=fopen("temp/".$_SESSION["id"]."p.html",'a');
	fwrite($myfile,	preg_replace( '/\\\"/','"', $item));
	fclose($myfile); 
}

?>
